import logging
import pathlib
import re
import textwrap
import typing

import agents
import jinja2
import openai
import pydantic
import rich
import rich.panel
import rich.text
from openai.types import ChatModel
from openai_usage import Usage
from rich_color_support import RichColorRotator
from str_or_none import str_or_none

__version__ = pathlib.Path(__file__).parent.joinpath("VERSION").read_text().strip()


logger = logging.getLogger(__name__)
console = rich.console.Console()
color_rotator = RichColorRotator()

DEFAULT_MODEL = "gpt-4.1-nano"

SUPPORTED_MODEL_TYPES: typing.TypeAlias = (
    agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel | ChatModel | str
)


class APSAgent:
    instructions_aps_j2: str = textwrap.dedent(
        """
        ## Role Instructions

        You are a Abstractive Proposition Segmentation (APS) Analyst.
        Your task is to read a text and extract specific rules, facts, notes or memories stated in the text.

        ## Abstractive Proposition Segmentation (APS)

        Each extracted fact MUST follow Abstractive Proposition Segmentation (APS) principles.
        APS is an analysis technique that breaks down text information into component parts.
        This means each fact should be a single, atomic proposition that captures one specific piece of information.

        ## Output Format Rules

        - Each extracted piece of information MUST be on a new line.
        - Each line MUST begin with the prefix `fact: `.
        - Extract the information as a concise statement following APS principles.
        - Each fact should contain only one atomic proposition or fact.
        - If you find NO facts, you MUST output exactly `fact: None`.
        - Do NOT include conversational filler, greetings, or your own explanations.
        - Only extract information from the text, not your own explanations.

        ## Examples

        ### Example 1

        text: ```
        TODO:
        ```

        analysis:
        fact: TODO:
        [DONE]

        ### Example 2

        text: ```
        TODO:
        ```

        analysis:
        fact: TODO:
        [DONE]

        ## Input Text

        text: ```
        {{ text }}
        ```

        analysis:
        """  # noqa: E501
    ).strip()

    async def run(
        self,
        text: str,
        *,
        model: typing.Optional[SUPPORTED_MODEL_TYPES] = None,
        tracing_disabled: bool = True,
        verbose: bool = False,
        console: rich.console.Console = console,
        color_rotator: RichColorRotator = color_rotator,
        width: int = 80,
        **kwargs,
    ) -> "APSResult":
        if not (sanitized_text := str_or_none(text)):
            raise ValueError("text is None")

        chat_model = self._to_chat_model(model)

        agent_instructions_template = jinja2.Template(self.instructions_aps_j2)
        user_input = agent_instructions_template.render(
            text=sanitized_text,
        )

        if verbose:
            __rich_panel = rich.panel.Panel(
                rich.text.Text(user_input),
                title="LLM INSTRUCTIONS",
                border_style=color_rotator.pick(),
                width=width,
            )
            console.print(__rich_panel)

        agent = agents.Agent(
            name="user-preferences-agent-analyze-language",
            model=chat_model,
            model_settings=agents.ModelSettings(temperature=0.0),
        )
        result = await agents.Runner.run(
            agent,
            user_input,
            run_config=agents.RunConfig(tracing_disabled=tracing_disabled),
        )
        usage = Usage.from_openai(result.context_wrapper.usage)
        usage.model = chat_model.model
        usage.cost = usage.estimate_cost(usage.model)

        if verbose:
            __rich_panel = rich.panel.Panel(
                rich.text.Text(str(result.final_output)),
                title="LLM OUTPUT",
                border_style=color_rotator.pick(),
                width=width,
            )
            console.print(__rich_panel)
            __rich_panel = rich.panel.Panel(
                rich.text.Text(usage.model_dump_json(indent=4)),
                title="LLM USAGE",
                border_style=color_rotator.pick(),
                width=width,
            )
            console.print(__rich_panel)

        return self._parse_aps_items(str(result.final_output))

    def _parse_aps_items(
        self,
        text: str,
    ) -> "APSResult":
        pattern = re.compile(r"^fact:\s*(.+)", re.MULTILINE | re.IGNORECASE)
        matches = pattern.findall(text)

        if len(matches) == 1 and matches[0].strip().lower() in ("none", "null"):
            return APSResult(input_text=text, facts=[])

        facts = [match.strip() for match in matches]
        return APSResult(input_text=text, facts=facts)

    def _to_chat_model(
        self,
        model: (
            agents.OpenAIChatCompletionsModel
            | agents.OpenAIResponsesModel
            | ChatModel
            | str
            | None
        ) = None,
    ) -> agents.OpenAIChatCompletionsModel | agents.OpenAIResponsesModel:
        model = DEFAULT_MODEL if model is None else model

        if isinstance(model, str):
            openai_client = openai.AsyncOpenAI()
            return agents.OpenAIResponsesModel(
                model=model,
                openai_client=openai_client,
            )

        else:
            return model


class APSResult(pydantic.BaseModel):
    input_text: str
    facts: typing.List[typing.Text]
    usages: typing.List[Usage] = pydantic.Field(default_factory=list)
