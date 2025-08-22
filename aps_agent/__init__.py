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

    instructions_facts_conflict_j2: str = textwrap.dedent(
        """
        ## Role Instructions

        You are a precise fact conflict analyzer. Identify contradictions, inconsistencies, and logical conflicts within provided facts. Check for direct contradictions, logical inconsistencies, timeline conflicts, numerical mismatches, and definitional conflicts. Be thorough but concise. Return "conflict: null" if no conflicts are found.

        ## Examples

        ### Example 1 (English — News)

        facts:
        - The earthquake struck at 3:47 AM local time on March 15, 2024
        - Emergency services received the first calls at 3:52 AM
        - The earthquake was detected by seismographs 2 minutes before it hit
        - No casualties were reported in the 7.2 magnitude earthquake

        conflict:
        - Timeline conflict: Seismographs cannot detect earthquake 2 minutes before it strikes (earthquakes are detected as they happen)
        - Logical inconsistency: Unlikely to have no casualties in a 7.2 magnitude earthquake in populated area
        [DONE]

        ### Example 2 (日本語 — 施設紹介)

        facts:
        - 東京スカイツリーは2012年に開業した
        - この施設は高さ634メートルの世界一高いタワーです
        - 展望台は地上450メートルの位置にあります
        - 建設は2008年に開始され、3年間で完成しました

        conflict:
        - 数値矛盾: 2008年開始で3年間なら2011年完成のはず（2012年開業と矛盾）
        - 事実誤認: 世界一高いタワーではない（ブルジュ・ハリファなど他にもっと高い建造物が存在）
        [DONE]

        ### Example 3 (中文 — 電子郵件)

        facts:
        - 會議將於下週二上午10點在會議室A舉行
        - 請所有部門主管準時參加此重要會議
        - 會議室A最多容納8人
        - 公司共有12個部門主管

        conflict:
        - 容量衝突: 會議室A僅能容納8人，但需要12位部門主管參加
        [DONE]

        ### Example 4 (한국어 — 전화 통화 메시지 내용)

        facts:
        - 민지가 오늘 오후 3시에 카페에서 만나자고 제안했다
        - 준호가 3시는 괜찮다고 동의했다
        - 만날 장소는 강남역 근처 스타벅스로 정했다
        - 민지가 먼저 도착해서 자리를 잡겠다고 말했다
        - 준호가 조금 늦을 수도 있다고 미리 양해를 구했다
        - 두 사람 모두 서로의 연락처를 알고 있다

        conflict: null
        [DONE]

        ## User Input Facts

        facts:
        {% for fact in facts -%}
        - {{ fact.fact }}
        {% endfor %}

        conflict:
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
        """Extract atomic facts from text using AI model.

        Returns structured facts following APS principles
        with usage information.
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

    async def detect_facts_conflict(
        self,
        facts: typing.List["Fact"],
        model: typing.Optional[SUPPORTED_MODEL_TYPES] = None,
        tracing_disabled: bool = True,
        verbose: bool = False,
        console: rich.console.Console = console,
        color_rotator: RichColorRotator = color_rotator,
        width: int = 80,
    ) -> "FactConflictResult":
        """Detect conflicts and inconsistencies in facts.

        Analyzes list of facts using AI model to identify
        contradictions and logical conflicts.
        """
        sanitized_facts = [
            Fact.model_validate_json(fact.model_dump_json()) for fact in facts
        ]
        if any(fact.fact == "" for fact in sanitized_facts):
            raise ValueError("Input facts contain empty content")

        chat_model = self._to_chat_model(model)

        agent_instructions_template = jinja2.Template(
            self.instructions_facts_conflict_j2
        )
        user_input = agent_instructions_template.render(
            facts=sanitized_facts,
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

        return FactConflictResult(
            input_facts=sanitized_facts,
            conflicts=self._parse_facts_conflict_items(str(result.final_output)),
            usages=[usage],
        )

    def _parse_aps_items(self, text: str) -> typing.List["Fact"]:
        """Parse AI output to extract individual facts.

        Extracts facts matching 'fact: ' pattern and returns
        them as Fact objects.
        """
        pattern = re.compile(r"^fact:\s*(.+)", re.MULTILINE | re.IGNORECASE)
        matches = pattern.findall(text)

        if len(matches) == 1 and matches[0].strip().lower() in ("none", "null"):
            return []

        return [Fact(fact=match.strip()) for match in matches]

    def _parse_facts_conflict_items(self, text: str) -> typing.List["FactConflict"]:
        """Parse AI output to extract conflicts.

        Extracts conflict items from AI response and returns
        them as FactConflict objects.
        """
        content: str = text if isinstance(text, str) else str(text)

        # Normalize non-empty lines for quick checks.
        nonempty_lines: typing.List[str] = [
            ln.strip() for ln in content.splitlines() if ln.strip()
        ]

        # Detect explicit "conflict: null|none" disclaimer.
        has_null_line: bool = any(
            re.match(r"^conflict:\s*(?:null|none)\s*$", ln, flags=re.IGNORECASE)
            is not None
            for ln in nonempty_lines
        )
        # Detect any other explicit conflict lines that carry payload.
        has_payload_conflict_line: bool = any(
            re.match(r"^conflict:\s*(.+)$", ln, flags=re.IGNORECASE) is not None
            and re.match(r"^conflict:\s*(?:null|none)\s*$", ln, flags=re.IGNORECASE)
            is None
            for ln in nonempty_lines
        )
        # Detect presence of bullet conflicts.
        has_bullets_anywhere: bool = any(
            re.match(r"^-\s+.+$", ln) is not None for ln in nonempty_lines
        )

        # If model declares "conflict: null|none" and provides nothing else, return [].
        if has_null_line and not has_payload_conflict_line and not has_bullets_anywhere:
            return []

        collected: typing.List[str] = []

        # 1) Collect explicit single-line 'conflict: ...' payloads (any string is OK).
        for ln in nonempty_lines:
            m: typing.Optional[re.Match[str]] = re.match(
                r"^conflict:\s*(.+)$", ln, flags=re.IGNORECASE
            )
            if m is None:
                continue
            payload: str = m.group(1).strip()
            if payload and payload.lower() not in {"null", "none"}:
                collected.append(payload)

        # 2) Collect any bullets beginning with "- " anywhere.
        for ln in nonempty_lines:
            mb: typing.Optional[re.Match[str]] = re.match(r"^-\s+(.*\S)\s*$", ln)
            if mb is None:
                continue
            bullet_text: str = mb.group(1).strip()
            if bullet_text:
                collected.append(bullet_text)

        # Deduplicate while preserving order; drop empty/null-like artifacts.
        seen: set[str] = set()
        unique_items: typing.List[str] = []
        for item in collected:
            normalized: str = re.sub(r"\s+", " ", item).strip()
            key: str = normalized.lower()
            if not normalized or key in {"null", "none"}:
                continue
            if key not in seen:
                seen.add(key)
                unique_items.append(normalized)

        return [FactConflict(conflict=conf) for conf in unique_items]

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
        """Convert input model to chat model instance.

        Handles different model types and converts them to
        OpenAI chat model instances.
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

    @pydantic.model_validator(mode="after")
    def validate_fact(self) -> typing.Self:
        self.fact = self.fact.strip()
        if self.fact == "":
            logger.warning("The fact content is empty")
        return self


class APSResult(pydantic.BaseModel):
    """Result container for APS analysis.

    Contains the original input text, extracted facts,
    and usage information from the AI model analysis.
    """

    input_text: str
    facts: typing.List[Fact]
    usages: typing.List[Usage] = pydantic.Field(default_factory=list)


class FactConflict(pydantic.BaseModel):
    """Represents a conflict found between facts.

    Contains details about contradictions or inconsistencies
    identified in analyzed facts.
    """

    conflict: str

    @pydantic.model_validator(mode="after")
    def validate_conflict(self) -> typing.Self:
        self.conflict = self.conflict.strip()
        if self.conflict == "":
            logger.warning("The conflict content is empty")
        return self


class FactConflictResult(pydantic.BaseModel):
    """Result container for fact conflict analysis.

    Contains input facts, identified conflicts, and
    usage information from AI model analysis.
    """

    input_facts: typing.List[Fact]
    conflicts: typing.List[FactConflict]
    usages: typing.List[Usage] = pydantic.Field(default_factory=list)
