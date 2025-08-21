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
    """Main agent for Abstractive Proposition Segmentation analysis.

    Extracts atomic facts from text using AI models and follows
    APS principles for structured information extraction.
    """

    instructions_aps_j2: str = textwrap.dedent(
        """
        ## Role Instructions

        You are a Abstractive Proposition Segmentation (APS) Analyst.
        Your task is to read a text and extract specific rules, facts, notes or memories stated in the text.

        ## Abstractive Proposition Segmentation (APS)

        Each extracted fact MUST follow Abstractive Proposition Segmentation (APS) principles.
        APS is an analysis technique that breaks down text information into component parts.
        This means each fact should be a single, atomic proposition that captures one specific piece of information.
        You will take a given passage of text and attempts to segment the content into the individual facts, statements, and ideas expressed in the text, and restates them in full sentences with small changes to the original text.

        ## Output Format Rules

        - Each extracted piece of information MUST be on a new line.
        - Each line MUST begin with the prefix `fact: `.
        - Extract the information as a concise statement following APS principles.
        - Each fact should contain only one atomic proposition or fact.
        - If you find NO facts, you MUST output exactly `fact: None`.
        - Do NOT include conversational filler, greetings, or your own explanations.
        - Only extract information from the text, not your own explanations.

        ## Examples

        ### Example 1 (English — News)

        text: ```
        On August 18, 2025, the Riverdale City Council approved a $5 million budget to expand protected bike lanes.
        The transportation department plans to add 12 kilometers of lanes across three districts.
        Construction is scheduled to begin in October 2025 and finish by July 2026, weather permitting.
        Monthly progress reports will be published on the city website.
        ```
        analysis:
        fact: The Riverdale City Council approved a $5 million bike-lane expansion budget.
        fact: The approval date was August 18, 2025.
        fact: The plan adds 12 kilometers of protected bike lanes.
        fact: The expansion covers three districts.
        fact: Construction is scheduled to begin in October 2025.
        fact: Construction is scheduled to finish by July 2026.
        fact: The transportation department will publish monthly progress reports on the city website.
        [DONE]

        ### Example 2 (日本語 — 施設紹介)

        text: ```
        新しく開業した「みなと図書館」は駅から徒歩3分の場所にあります。
        開館時間は9:00〜20:00で、毎週火曜日が休館日です。
        自習席は60席あり、先着順で利用できます。
        メーカーA製の3Dプリンタを予約制で提供しており、1回の利用は最大2時間です。
        託児スペースは10:00〜16:00の間に利用でき、対象は1〜6歳です。
        館内では無料Wi-Fiを利用できます。
        ```
        analysis:
        fact: みなと図書館は駅から徒歩3分の場所にある。
        fact: 開館時間は9:00から20:00までである。
        fact: 休館日は毎週火曜日である。
        fact: 自習席は60席あり先着順である。
        fact: 3Dプリンタは予約制で1回最大2時間利用できる。
        fact: 託児スペースは10:00から16:00に利用できる。
        fact: 託児スペースの対象年齢は1歳から6歳である。
        fact: 館内では無料Wi-Fiが提供されている。
        [DONE]

        ### Example 3 (中文 — 電子郵件)

        text: ```
        主旨：本週會議改期與資料更新
        各位好，本週產品評審會議將從週四改到週五上午10:00（UTC+8）。
        地點從A棟3樓會議室改為線上會議；連結會在會前30分鐘寄出。
        請於週四18:00前回覆是否出席；若需代理出席請註明姓名。
        會議資料夾已更新至最新版簡報與議程（/Team Drive/Reviews/2025-08-Week3）。
        ```
        analysis:
        fact: 產品評審會議從週四改到週五上午10:00（UTC+8）。
        fact: 會議地點改為線上會議。
        fact: 線上會議連結將在會前30分鐘寄出。
        fact: 與會者須在週四18:00前回覆是否出席。
        fact: 需要代理出席者必須提供代理人姓名。
        fact: 會議資料夾包含最新版簡報與議程。
        fact: 會議資料夾路徑為 /Team Drive/Reviews/2025-08-Week3。
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
        """Process text using AI model to extract atomic facts.

        Analyzes input text and returns structured facts following
        APS principles with usage information.

        Args:
            text: Input text to analyze
            model: AI model to use for analysis
            tracing_disabled: Whether to disable tracing
            verbose: Whether to show detailed output
            console: Rich console for output display
            color_rotator: Color rotator for styling
            width: Display width for panels
            **kwargs: Additional arguments

        Returns:
            APSResult containing extracted facts and usage info
        """
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

        return APSResult(
            input_text=text,
            facts=self._parse_aps_items(str(result.final_output)),
            usages=[usage],
        )

    def _parse_aps_items(self, text: str) -> typing.List["Fact"]:
        """Parse AI output to extract individual facts.

        Extracts facts from text that match the 'fact: ' pattern
        and returns them as Fact objects.

        Args:
            text: Raw text output from AI model

        Returns:
            List of Fact objects extracted from the text
        """
        pattern = re.compile(r"^fact:\s*(.+)", re.MULTILINE | re.IGNORECASE)
        matches = pattern.findall(text)

        if len(matches) == 1 and matches[0].strip().lower() in ("none", "null"):
            return []

        return [Fact(fact=match.strip()) for match in matches]

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
        """Convert input model to appropriate chat model instance.

        Handles different model input types and converts them to
        OpenAI chat model instances for use with the agents library.

        Args:
            model: Input model (string, ChatModel, or OpenAI model instance)

        Returns:
            Configured OpenAI chat model ready for use
        """
        model = DEFAULT_MODEL if model is None else model

        if isinstance(model, str):
            openai_client = openai.AsyncOpenAI()
            return agents.OpenAIResponsesModel(
                model=model,
                openai_client=openai_client,
            )

        else:
            return model


class Fact(pydantic.BaseModel):
    """Represents a single atomic fact extracted from text.

    Each fact is an atomic proposition that captures one specific
    piece of information following APS principles.
    """

    fact: str


class APSResult(pydantic.BaseModel):
    """Result container for APS analysis.

    Contains the original input text, extracted facts,
    and usage information from the AI model analysis.
    """

    input_text: str
    facts: typing.List[Fact]
    usages: typing.List[Usage] = pydantic.Field(default_factory=list)
