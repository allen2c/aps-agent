# APS Agent

[![PyPI version](https://img.shields.io/pypi/v/aps-agent.svg)](https://pypi.org/project/aps-agent/)
[![Python Version](https://img.shields.io/pypi/pyversions/aps-agent.svg)](https://pypi.org/project/aps-agent/)
[![License](https://img.shields.io/pypi/l/aps-agent.svg)](https://opensource.org/licenses/MIT)

An intelligent agent for **Abstractive Proposition Segmentation (APS)** analysis that extracts atomic facts from text and detects logical conflicts between facts using AI models.

## Features

- **Atomic Fact Extraction**: Break down text into individual, atomic propositions following APS principles
- **Multi-language Support**: Process text in English, Japanese, Chinese, and Korean
- **Conflict Detection**: Identify logical contradictions and inconsistencies between facts
- **AI-Powered Analysis**: Leverages OpenAI models for intelligent text analysis
- **Structured Output**: Returns well-typed results with usage information and cost estimation
- **Comprehensive Validation**: Built-in data validation and error handling
- **Rich CLI Output**: Optional verbose mode with beautiful console output using Rich

## Installation

```bash
pip install aps-agent
```

### Requirements

- Python 3.11 or higher
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### From Source

```bash
git clone https://github.com/allen2c/aps-agent.git
cd aps-agent
pip install -e .
```

## Quick Start

### Basic Usage

```python
import asyncio
from aps_agent import APSAgent

async def main():
    # Initialize the agent
    agent = APSAgent()

    # Extract facts from text
    text = """
    The Riverdale City Council approved a $5 million budget for bike lanes on August 18, 2025.
    Construction will begin in October 2025 and finish by July 2026, weather permitting.
    """

    result = await agent.run(text, verbose=True)

    print(f"Input text: {result.input_text}")
    print(f"Extracted {len(result.facts)} facts:")
    for fact in result.facts:
        print(f"  - {fact.fact}")

asyncio.run(main())
```

Output:

```plaintext
Input text: The Riverdale City Council approved a $5 million budget for bike lanes on August 18, 2025. Construction will begin in October 2025 and finish by July 2026, weather permitting.
Extracted 6 facts:
  - The Riverdale City Council approved a bike-lane expansion budget on August 18, 2025.
  - The approved budget amount is $5 million.
  - Construction is scheduled to begin in October 2025.
  - Construction is scheduled to finish by July 2026.
  - Construction completion depends on weather conditions.
  - The transportation department will publish monthly progress reports.
```

### Fact Conflict Detection

```python
import asyncio
from aps_agent import APSAgent, Fact

async def main():
    agent = APSAgent()

    # Create facts to analyze for conflicts
    facts = [
        Fact(fact="The meeting is scheduled for 2:00 PM today"),
        Fact(fact="The meeting is scheduled for 3:00 PM today"),
        Fact(fact="John will attend the meeting"),
        Fact(fact="John is currently in Tokyo"),
        Fact(fact="John is currently in London"),
    ]

    # Detect conflicts
    conflict_result = await agent.detect_facts_conflict(facts, verbose=True)

    if conflict_result.conflicts:
        print("Conflicts found:")
        for conflict in conflict_result.conflicts:
            print(f"  - {conflict.conflict}")
    else:
        print("No conflicts detected")

asyncio.run(main())
```

## Configuration

### Model Selection

You can specify different OpenAI models:

```python
# Use GPT-4
result = await agent.run(text, model="gpt-4")

# Use GPT-4o
result = await agent.run(text, model="gpt-4o")

# Use a custom model string
result = await agent.run(text, model="gpt-4.1-nano")
```

### Verbose Mode

Enable verbose mode to see detailed AI interactions:

```python
result = await agent.run(
    text,
    verbose=True,  # Shows instructions, AI output, and usage info
    width=100     # Adjust console width for better display
)
```

### Tracing

Control AI tracing for debugging:

```python
result = await agent.run(
    text,
    tracing_disabled=False  # Enable tracing for debugging
)
```

## API Reference

### APSAgent

Main class for APS analysis.

#### Methods

- `run(text, model=None, tracing_disabled=True, verbose=False, console=None, color_rotator=None, width=80, **kwargs)`: Extract atomic facts from text
- `detect_facts_conflict(facts, model=None, tracing_disabled=True, verbose=False, console=None, color_rotator=None, width=80)`: Detect conflicts between facts

#### Parameters

- `text` (str): Input text to analyze (required for `run()`)
- `facts` (List[Fact]): List of facts to analyze for conflicts (required for `detect_facts_conflict()`)
- `model` (Optional[str]): OpenAI model to use (default: "gpt-4.1-nano")
- `tracing_disabled` (bool): Whether to disable AI tracing (default: True)
- `verbose` (bool): Enable verbose output with rich formatting (default: False)
- `console` (Console): Rich console instance for output (default: built-in console)
- `color_rotator` (RichColorRotator): Color rotator for output styling (default: built-in rotator)
- `width` (int): Console width for display (default: 80)

### Data Models

#### Fact

Represents a single atomic fact.

```python
Fact(fact: str)
```

#### APSResult

Result container for APS analysis.

```python
APSResult(
    input_text: str,
    facts: List[Fact],
    usages: List[Usage]
)
```

#### FactConflict

Represents a conflict between facts.

```python
FactConflict(conflict: str)
```

#### FactConflictResult

Result container for conflict analysis.

```python
FactConflictResult(
    input_facts: List[Fact],
    conflicts: List[FactConflict],
    usages: List[Usage]
)
```

## Abstractive Proposition Segmentation (APS)

APS is an analysis technique that breaks down text information into atomic components. Each extracted fact must follow these principles:

### Core Rules

1. **Atomic Principle**: Each fact contains ONLY ONE piece of information
2. **No Duplication**: Extract each unique piece of information only once
3. **Direct Information Only**: Extract only what's directly stated in the text
4. **Precise Attribution**: Maintain clear attribution to speakers when applicable

### Output Format

Each extracted fact follows the format:

```plaintext
fact: [single atomic proposition]
```

Example:

```plaintext
text: "The meeting starts at 2:00 PM and will last 1 hour."
facts:
  - fact: The meeting starts at 2:00 PM
  - fact: The meeting will last 1 hour
```

## Examples

### Multi-language Support

The agent supports multiple languages with the same API:

```python
# English
text_en = "The company reported $10 million in quarterly profits."

# Japanese
text_ja = "会社は四半期利益として1,000万円を報告した。"

# Chinese
text_zh = "公司报告季度利润为1000万元人民币。"

# Korean
text_ko = "회사는 분기 수익으로 1,000만원을 보고했다。"
```

### Complex Text Analysis

```python
complex_text = """
Apple Inc. announced record-breaking iPhone 15 sales in Q4 2024, with 80 million units sold worldwide.
The company's revenue reached $119.6 billion, representing a 2.8% year-over-year increase.
However, supply chain constraints limited production to 85 million units during the quarter.
The CEO stated that the company is working to resolve manufacturing bottlenecks by Q2 2025.
"""

result = await agent.run(complex_text, verbose=True)
```

This will extract atomic facts about sales figures, revenue, constraints, and company statements as separate, individual propositions.

## Error Handling

The agent includes comprehensive error handling:

```python
try:
    result = await agent.run(text)
except ValueError as e:
    print(f"Input validation error: {e}")
except Exception as e:
    print(f"Analysis error: {e}")
```

Common errors:

- Empty or None text input
- Invalid fact content
- API communication errors
- Model availability issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Setup

```bash
git clone https://github.com/allen2c/aps-agent.git
cd aps-agent
pip install -e ".[dev]"
pytest
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/allen2c/aps-agent).
