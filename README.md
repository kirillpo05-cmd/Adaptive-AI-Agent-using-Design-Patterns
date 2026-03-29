# Adaptive AI Agent using Design Patterns

A command-line personal assistant built on the **Google Gemini API** that autonomously reasons about user requests and dispatches to external tools.
The entire implementation lives in a single file: `AI_agent.py`.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting a Gemini API Key](#getting-a-gemini-api-key)
- [Running the Agent](#running-the-agent)
- [Available Tools](#available-tools)
- [CLI Commands](#cli-commands)
- [Example Session](#example-session)
- [Architecture](#architecture)
- [Design Patterns Applied](#design-patterns-applied)
- [Error Handling](#error-handling)
- [Adding a New Tool](#adding-a-new-tool)

---

## Features

- Conversational AI agent with multi-turn memory
- ReAct reasoning loop (Reason → Act → Observe)
- 6 built-in tools: calculator, clock, weather, currency, translation, file reader
- Colour-coded console logging via the Observer pattern
- Clean SOLID architecture — adding a new tool requires zero changes to core classes

---

## Requirements

- Python 3.10 or newer (tested on 3.12)
- Internet connection (for weather, currency, translation, and Gemini API calls)
- A Google Gemini API key (free tier available)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Adaptive-AI-Agent-using-Design-Patterns.git
cd Adaptive-AI-Agent-using-Design-Patterns
```

### 2. Install Python dependencies

```bash
# Using py launcher (Windows)
py -3.12 -m pip install requests google-genai

# Or using pip directly
pip install requests google-genai
```

---

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com)
2. Sign in with your Google account
3. Click **Get API key** → **Create API key**
4. Copy the generated key

> The free tier provides a generous quota for personal use.
> If you see a `429 RESOURCE_EXHAUSTED` error, your key's daily quota has been reached — wait 24 hours or enable billing.

---

## Running the Agent

### Option A — Set API key via environment variable (recommended)

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"
py -3.12 AI_agent.py
```

```bash
# Linux / macOS
export GEMINI_API_KEY="your-api-key-here"
python3 AI_agent.py
```

### Option B — Hardcode the key in the file

Open `AI_agent.py` and find line 723:

```python
api_key = os.environ.get("GEMINI_API_KEY", "your-api-key-here")
```

Replace `your-api-key-here` with your actual key.

Then run:

```bash
py -3.12 AI_agent.py
```

### Expected startup output

```
+----------------------------------------------------------+
|          Adaptive AI Agent  -  Design Patterns           |
|  Tools: calculator, time, weather, currency,             |
|         translate, read_file                             |
|  Commands: 'quit'/'exit' to stop, 'clear' to reset       |
+----------------------------------------------------------+

Agent ready. Type your message below.

You:
```

---

## Available Tools

| # | Tool name | Description | External API |
|---|---|---|---|
| 1 | `calculator` | Evaluates arithmetic expressions (`+`, `-`, `*`, `/`, `**`, `sqrt`, `sin`, etc.) | Python `math` (no network) |
| 2 | `get_current_time` | Returns the current local date and time | `datetime` (no network) |
| 3 | `get_weather` | Current weather conditions for any city | Open-Meteo (free, no key) |
| 4 | `convert_currency` | Live currency conversion between any two ISO 4217 currencies | Frankfurter (free, no key) |
| 5 | `translate_text` | Translates text between languages using ISO 639-1 codes | MyMemory (free, no key) |
| 6 | `read_local_file` | Reads a local text file and returns its contents to the agent | Local filesystem |

---

## CLI Commands

| Input | Effect |
|---|---|
| Any natural language | Processed by the agent |
| `clear` | Resets conversation history |
| `quit` or `exit` | Exits the program |
| `Ctrl+C` | Force exit |

---

## Example Session

```
You: What is 15% of 847?
[TOOL CALL] calculator({"expression": "0.15 * 847"})
[TOOL RESULT] calculator -> 127.05
Agent: 15% of 847 is 127.05.

You: What's the weather in Tokyo?
[TOOL CALL] get_weather({"city": "Tokyo"})
[TOOL RESULT] get_weather -> Weather in Tokyo, Japan: Partly cloudy, Temperature: 18C, Wind: 12 km/h, Humidity: 65%
Agent: It is currently partly cloudy in Tokyo with a temperature of 18°C and wind at 12 km/h.

You: How much is 100 USD in EUR?
[TOOL CALL] convert_currency({"amount": 100, "from_currency": "USD", "to_currency": "EUR"})
[TOOL RESULT] convert_currency -> 100 USD = 91.82 EUR (rate as of 2025-03-29)
Agent: 100 US dollars is approximately 91.82 euros.

You: Translate "Good morning" to Japanese
[TOOL CALL] translate_text({"text": "Good morning", "source_language": "en", "target_language": "ja"})
[TOOL RESULT] translate_text -> Translation (en -> ja): "おはようございます"
Agent: "Good morning" in Japanese is おはようございます (Ohayou gozaimasu).

You: What time is it?
[TOOL CALL] get_current_time({})
[TOOL RESULT] get_current_time -> Current local date/time: Sunday, March 29 2025  14:35:22 (timezone: local)
Agent: The current local time is 2:35 PM on Sunday, March 29, 2025.

You: clear
Conversation history cleared.

You: exit
Goodbye!
```

---

## Architecture

The system is split into five focused components:

```
User Input
    │
    ▼
┌─────────────┐     history      ┌───────────────┐
│    Agent    │◄────────────────►│ MemoryManager │
│ (ReAct loop)│                  └───────────────┘
└──────┬──────┘
       │ tool dispatch
       ▼
┌──────────────┐   register/lookup   ┌──────────────┐
│ ToolRegistry │◄────────────────────│  BaseTool(s) │
└──────────────┘                     └──────────────┘
       │ events
       ▼
┌───────────────┐
│ ConsoleLogger │  (Observer)
└───────────────┘
```

| Class | Responsibility | Pattern |
|---|---|---|
| `BaseTool` | Abstract tool interface | OCP / DIP |
| `ToolRegistry` | Register, look up, and execute tools | Factory / Registry |
| `MemoryManager` | Store and replay conversation history | SRP |
| `Agent` | Drive the ReAct reasoning loop | Strategy |
| `ConsoleLogger` | Log tool events to the console | Observer |

---

## Design Patterns Applied

**ReAct (Reason → Act → Observe)**
The agent sends the conversation to Gemini, receives either a tool call or a final answer, executes the tool, appends the result, and repeats — up to `MAX_REACT_STEPS = 10` iterations.

**Strategy Pattern**
Each tool is an interchangeable strategy. `ToolRegistry.execute()` selects the correct strategy at runtime; the `Agent` never references concrete tool classes.

**Factory / Registry Pattern**
`ToolRegistry` acts as a factory registry. Tools are registered at startup in `build_agent()`; the rest of the system only knows about the `BaseTool` interface.

**Observer Pattern**
`AgentObserver` defines the event interface. `ConsoleLogger` is the concrete observer. The `Agent` calls `_notify_*` methods without knowing what observers are attached.

**Open/Closed Principle (OCP)**
`BaseTool` is the extension point. To add a new tool: subclass `BaseTool`, implement `name`, `execute()`, and `get_declaration()`, then register it — no changes to `Agent` or `ToolRegistry` are needed.

**Dependency Inversion Principle (DIP)**
`Agent` depends on the abstract `BaseTool` interface and `ToolRegistry`, not on any concrete tool implementation.

**Single Responsibility Principle (SRP)**
Each class has exactly one reason to change: `MemoryManager` owns history, `ToolRegistry` owns tool dispatch, `Agent` owns the reasoning loop.

---

## Error Handling

| Scenario | How it is handled |
|---|---|
| API error (network, quota, auth) | Caught in `Agent.chat()`; friendly message returned; conversation continues |
| Invalid tool arguments | `ToolRegistry` catches `TypeError`; error string fed back to the LLM |
| Unknown tool name | Registry returns an explanatory error string; loop continues |
| Tool runtime exception | All exceptions inside `BaseTool.execute()` are caught and returned as strings |
| Infinite reasoning loop | `MAX_REACT_STEPS = 10` caps iterations; fallback message returned |

---

## Adding a New Tool

1. Subclass `BaseTool` and implement the three required members:

```python
class JokeTool(BaseTool):

    @property
    def name(self):
        return "get_joke"

    def execute(self, category="general"):
        return "Why do programmers prefer dark mode? Because light attracts bugs."

    def get_declaration(self):
        return {
            "name": self.name,
            "description": "Tell a joke.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "category": {"type": "STRING", "description": "Joke category."}
                },
                "required": [],
            },
        }
```

2. Register it in `build_agent()`:

```python
registry.register(JokeTool())
```

No other changes required.
