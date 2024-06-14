import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_experimental.pydantic_v1 import BaseModel, Field
from langfuse.callback import CallbackHandler
from .memory import GenerativeAgentMemory
langfuse_handler = CallbackHandler()

def remove_trailing_commas(json_str):
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*\]', ']', json_str)
    return json_str

jikken = "The following are instructions for the output format. Please always follow the instructions and do not write any extra characters."

class GenerativeAgent(BaseModel):
    name: str
    age: Optional[int] = None
    traits: str = "N/A"
    status: str
    memory: GenerativeAgentMemory
    llm: BaseLanguageModel
    verbose: bool = False
    summary: str = ""
    summary_refresh_seconds: int = 3600
    last_refreshed: datetime = Field(default_factory=datetime.now)
    daily_summaries: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory, callbacks=[langfuse_handler]
        )

    def _get_entity_from_observation(self, observation: str) -> str:
        response_schemas = [
            ResponseSchema(name="entity", description="Agent or entity in the observation"),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        method_prompt = (
            "What is the observed entity in the following observation? \nobservation: {observation}"
        )

        prompt = PromptTemplate(
            template = method_prompt+"\n"+jikken+"\n{format_instructions}",
            input_variables=["observation"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        result = parser.parse(remove_trailing_commas(self.chain(prompt).run(observation=observation).strip()))
        return result['entity']
    
    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        response_schemas = [
            ResponseSchema(name="action", description="Action of the entity in the observation. Start writing from the verb, without including the entity."),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        method_prompt = (
            "What is the {entity} doing in the following observation? \nobservation: {observation}"
        )

        prompt = PromptTemplate(
            template = method_prompt+"\n"+jikken+"\n{format_instructions}",
            input_variables=["observation", "entity"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        result = parser.parse(remove_trailing_commas(self.chain(prompt).run(entity=entity_name, observation=observation).strip()))
        return result['action']

    def summarize_related_memories(self, observation: str) -> str:
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)

        relationship_description = f"Relationship between {self.name} and {entity_name}"
        response_schemas = [
            ResponseSchema(name="relationship", description=f"{relationship_description}"),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        method_prompt = (
            """
{q1}?
Context from memory:
{relevant_memories}
Relevant context: 
"""
        )

        prompt = PromptTemplate(
            template = method_prompt+"\n"+jikken+"\n{format_instructions}",
            input_variables=["q1", "relevant_memories"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = f"{entity_name} {entity_action}"

        result = parser.parse(remove_trailing_commas(self.chain(prompt).run(q1=q1, queries=[q1, q2]).strip()))
        return result['relationship']

    def _generate_reaction(
        self, observation: str, suffix: str, format_instructions: str, now: Optional[datetime] = None
    ) -> str:
        method_prompt = (
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nMost recent observations: {most_recent_memories}"
            + "\nObservation: {observation}"
            + "\n\n"
        )

        prompt = PromptTemplate(
            template = method_prompt+suffix,
            input_variables=[
                "agent_summary_description", 
                "current_time", 
                "relevant_memories", 
                "agent_name",
                "observation", 
                "agent_status",
                "format_instructions"],
            )
        
        agent_summary_description = self.get_summary(now=now)
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            observation=observation,
            agent_status=self.status,
            format_instructions=format_instructions
        )

        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(most_recent_memories="", **kwargs)
        )
        kwargs[self.memory.most_recent_memories_token_key] = consumed_tokens
        return remove_trailing_commas(self.chain(prompt=prompt).run(**kwargs).strip())

    def generate_reaction(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        name = self.name
        response_schemas = [
            ResponseSchema(name="type", description="Type of reaction of the agent. Choose REACT or SAY or NOTHING."),
            ResponseSchema(name="content", description=f"The content of the {name}'s reaction."),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        method_prompt = (
        "Should {agent_name} react to the observation, and if so, "
        "what would be an appropriate reaction?"
        "\n\n"
    )

        prompt = (
            method_prompt +"\n"+jikken+"\n{format_instructions}"
        )

        full_result = self._generate_reaction(
            observation, prompt, format_instructions=parser.get_format_instructions(), now=now
        )
        result = parser.parse(full_result)
        self.memory.save_context(
            {},
            {
                self.memory.add_memory_key: f"{self.name} observed "
                f"{observation} and reacted by {result['content']}",
                self.memory.now_key: now,
            },
        )
        if result['type'].upper() == "REACT":
            reaction = result['content']
            return False, f"{self.name} {reaction}"
        if result['type'].upper() == "SAY":
            said_value = result['content']
            return True, f"{self.name} said {said_value}"
        else:
            return False, result['content']


    def generate_dialogue_response(
        self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        name = self.name
        response_schemas = [
            ResponseSchema(name="type", description=f"Type of dialogue response of {name}. Choose 'SAY' to continue the conversation, or 'GOODBYE' to end it."),
            ResponseSchema(name="content", description="Content of the dialogue response."),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        method_prompt = (
            "What would {agent_name} say? To end the conversation, write:"
            ' GOODBYE: "what to say". Otherwise to continue the conversation,'
            ' write: SAY: "what to say next"\n\n'
        )

        prompt = (
            method_prompt +"\n"+jikken+"\n{format_instructions}"
        )

        full_result = self._generate_reaction(
            observation, prompt, format_instructions=parser.get_format_instructions(), now=now
        )
        result = parser.parse(full_result)
        if result['type'].upper() == "GOODBYE":
            farewell = result['content']
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {farewell}",
                    self.memory.now_key: now,
                },
            )
            return False, f"{self.name} said {farewell}"
        if result['type'].upper() == "SAY":
            response_text = result['content']
            self.memory.save_context(
                {},
                {
                    self.memory.add_memory_key: f"{self.name} observed "
                    f"{observation} and said {response_text}",
                    self.memory.now_key: now,
                },
            )
            return True, f"{self.name} said {response_text}"
        else:
            return False, result


    def _compute_agent_summary(self) -> str:
        response_schemas = [
            ResponseSchema(name="summary", description="Summary of the agent's core characteristics."),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        method_prompt = (
            "Do not embellish. How would you summarize {name}'s core characteristics based on the"
            + " following statements:\n"
            + "{relevant_memories}"
        )

        prompt = PromptTemplate(
            template = method_prompt+"\n"+jikken+"\n{format_instructions}",
            input_variables=["name", "relevant_memories"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        result = parser.parse(remove_trailing_commas(self.chain(prompt).run(name=self.name, queries=[f"{self.name}'s core characteristics"]).strip()))
        return (
            result['summary']
        )

    def get_summary(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
            not self.summary
            or since_refresh >= self.summary_refresh_seconds
            or force_refresh
        ):
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        age = self.age if self.age is not None else "N/A"
        return (
            f"Name: {self.name} (age: {age})"
            + f"\nInnate traits: {self.traits}"
            + f"\n{self.summary}"
        )


    def get_full_header(
        self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )