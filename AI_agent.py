"""
Adaptive AI Agent using Design Patterns
========================================
Architecture:
  - BaseTool          : Abstract interface (OCP / DIP)
  - ToolRegistry      : Factory / Registry pattern
  - MemoryManager     : Conversation history (SRP)
  - Agent             : ReAct loop orchestrator (Strategy pattern for tool dispatch)
  - Observer / Logger : Lightweight observer for event logging (bonus)

Tools implemented (6 total):
  1. CalculatorTool   - evaluate arithmetic expressions
  2. TimeTool         - current date / time
  3. WeatherTool      - current weather via Open-Meteo (no key required)
  4. CurrencyTool     - currency conversion via frankfurter.app (no key required)
  5. TranslationTool  - translate text via MyMemory (custom, no key required)
  6. FileReaderTool   - read a local text file (custom)

Requirements:
  py -3.11 -m pip install google-genai requests

Run:
  py -3.11 AI_agent.py
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard library
# ──────────────────────────────────────────────────────────────────────────────
import os
import math
import json
import datetime
import traceback
from abc import ABC, abstractmethod

# ──────────────────────────────────────────────────────────────────────────────
# Third-party
# ──────────────────────────────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: py -3.11 -m pip install requests")

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise SystemExit("Missing dependency: py -3.11 -m pip install google-genai")


# ==============================================================================
# OBSERVER PATTERN  (Bonus)
# ==============================================================================

class AgentObserver(ABC):
    """Abstract observer notified on agent lifecycle events."""

    @abstractmethod
    def on_tool_call(self, tool_name, args):
        pass

    @abstractmethod
    def on_tool_result(self, tool_name, result):
        pass

    @abstractmethod
    def on_error(self, context, error):
        pass


class ConsoleLogger(AgentObserver):
    """Concrete observer — prints structured events to stdout."""

    _CYAN  = "\033[96m"
    _GREEN = "\033[92m"
    _RED   = "\033[91m"
    _RESET = "\033[0m"

    def on_tool_call(self, tool_name, args):
        print("{}[TOOL CALL] {}({}){}".format(
            self._CYAN, tool_name, json.dumps(args, ensure_ascii=False), self._RESET))

    def on_tool_result(self, tool_name, result):
        print("{}[TOOL RESULT] {} -> {}{}".format(
            self._GREEN, tool_name, result, self._RESET))

    def on_error(self, context, error):
        print("{}[ERROR] {}: {}{}".format(self._RED, context, error, self._RESET))


# ==============================================================================
# BASE TOOL  (OCP / DIP)
# ==============================================================================

class BaseTool(ABC):
    """
    Abstract interface every tool must implement.

    Adding a new tool requires only:
      1. Subclass BaseTool
      2. Implement name, execute(), get_declaration()
      3. Register it — no Agent or ToolRegistry changes needed (OCP).
    """

    @property
    @abstractmethod
    def name(self):
        """Unique identifier used by the LLM to invoke this tool."""

    @abstractmethod
    def execute(self, **kwargs):
        """Run the tool; return a string observation for the LLM."""

    @abstractmethod
    def get_declaration(self):
        """
        Return a dict: { "name": ..., "description": ..., "parameters": {...} }
        Parameters must follow JSON Schema with UPPERCASE type values
        (e.g. "OBJECT", "STRING", "NUMBER") as required by the Gemini API.
        """


# ==============================================================================
# TOOL REGISTRY  (Factory / Registry Pattern)
# ==============================================================================

class ToolRegistry:
    """
    Central registry that maps tool names to BaseTool instances.

    Responsibilities (SRP):
      - Register tools
      - Look up tools by name
      - Execute a named tool with provided arguments
      - Build the Gemini Tool spec for the API call

    The Agent never instantiates tools directly; it always goes through the
    registry, decoupling tool creation from tool use (Factory pattern).
    """

    def __init__(self):
        self._tools = {}

    def register(self, tool):
        if not isinstance(tool, BaseTool):
            raise TypeError("Expected a BaseTool instance, got {}".format(type(tool)))
        self._tools[tool.name] = tool

    def get(self, name):
        return self._tools.get(name)

    def available_names(self):
        return list(self._tools.keys())

    def execute(self, name, args):
        """
        Execute the named tool. Returns an error string on failure (never raises),
        so the Agent loop can feed the failure back to the LLM as an observation.
        """
        tool = self._tools.get(name)
        if tool is None:
            return "Error: unknown tool '{}'. Available: {}".format(
                name, self.available_names())
        try:
            return tool.execute(**args)
        except TypeError as exc:
            return "Error: invalid arguments for tool '{}': {}".format(name, exc)
        except Exception as exc:
            return "Error: tool '{}' raised an unexpected error: {}".format(name, exc)

    def build_gemini_tool(self):
        """Convert all registered tools into a single types.Tool object."""
        declarations = []
        for tool in self._tools.values():
            d = tool.get_declaration()
            declarations.append(
                types.FunctionDeclaration(
                    name=d["name"],
                    description=d["description"],
                    parameters=d.get("parameters"),
                )
            )
        return types.Tool(function_declarations=declarations)


# ==============================================================================
# MEMORY MANAGER  (SRP)
# ==============================================================================

class MemoryManager:
    """
    Owns the conversation history for a single session.

    Stores turns as types.Content objects so they can be passed directly
    to the Gemini API without conversion.
    """

    def __init__(self):
        self._history = []

    def add_content(self, content):
        """Add a types.Content object directly (used for model turns with tool calls)."""
        self._history.append(content)

    def add_user_text(self, text):
        self._history.append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )

    def add_model_text(self, text):
        self._history.append(
            types.Content(role="model", parts=[types.Part(text=text)])
        )

    def get_history(self):
        return list(self._history)

    def clear(self):
        self._history.clear()

    def __len__(self):
        return len(self._history)


# ==============================================================================
# CONCRETE TOOLS  (Strategy Pattern — interchangeable algorithms)
# ==============================================================================

# ── 1. Calculator ─────────────────────────────────────────────────────────────

class CalculatorTool(BaseTool):
    """Safely evaluate arithmetic expressions using Python's math module."""

    @property
    def name(self):
        return "calculator"

    def execute(self, expression=""):
        if not expression:
            return "Error: no expression provided."
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round})
        try:
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        except ZeroDivisionError:
            return "Error: division by zero."
        except Exception as exc:
            return "Error evaluating '{}': {}".format(expression, exc)

    def get_declaration(self):
        return {
            "name": self.name,
            "description": (
                "Evaluate a mathematical expression. Supports arithmetic operators "
                "(+, -, *, /, **) and Python math functions (sin, cos, sqrt, log, etc.)."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "expression": {
                        "type": "STRING",
                        "description": "A Python arithmetic expression, e.g. '2**10 + sqrt(16)'.",
                    }
                },
                "required": ["expression"],
            },
        }


# ── 2. Time ───────────────────────────────────────────────────────────────────

class TimeTool(BaseTool):
    """Return the current local date and time."""

    @property
    def name(self):
        return "get_current_time"

    def execute(self, timezone="local"):
        now = datetime.datetime.now()
        return "Current local date/time: {} (timezone: {})".format(
            now.strftime("%A, %B %d %Y  %H:%M:%S"), timezone)

    def get_declaration(self):
        return {
            "name": self.name,
            "description": "Return the current local date and time.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "timezone": {
                        "type": "STRING",
                        "description": "Timezone label (informational), e.g. 'UTC'. Defaults to 'local'.",
                    }
                },
                "required": [],
            },
        }


# ── 3. Weather ────────────────────────────────────────────────────────────────

class WeatherTool(BaseTool):
    """
    Fetch current weather for a city using the free Open-Meteo API
    (no API key required). Geocoding via the Open-Meteo geocoding endpoint.
    """

    _GEO_URL     = "https://geocoding-api.open-meteo.com/v1/search"
    _WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    @property
    def name(self):
        return "get_weather"

    def execute(self, city=""):
        if not city:
            return "Error: city name is required."

        try:
            geo = requests.get(
                self._GEO_URL,
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=8,
            ).json()
        except requests.RequestException as exc:
            return "Error: could not reach geocoding service: {}".format(exc)

        results = geo.get("results")
        if not results:
            return "Error: city '{}' not found.".format(city)

        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        display_name = "{}, {}".format(loc.get("name", city), loc.get("country", ""))

        try:
            wx = requests.get(
                self._WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": True,
                    "hourly": "relativehumidity_2m",
                    "forecast_days": 1,
                    "timezone": "auto",
                },
                timeout=8,
            ).json()
        except requests.RequestException as exc:
            return "Error: could not reach weather service: {}".format(exc)

        cw = wx.get("current_weather", {})
        if not cw:
            return "Error: weather data unavailable for '{}'.".format(city)

        humidity = "N/A"
        hourly = wx.get("hourly", {})
        times = hourly.get("time", [])
        humidities = hourly.get("relativehumidity_2m", [])
        if times and humidities:
            current_hour = datetime.datetime.now().strftime("%Y-%m-%dT%H:00")
            if current_hour in times:
                idx = times.index(current_hour)
                humidity = "{}%".format(humidities[idx])

        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog",
            51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
            61: "Light rain", 63: "Rain", 65: "Heavy rain",
            71: "Light snow", 73: "Snow", 75: "Heavy snow",
            80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail",
        }
        code = cw.get("weathercode", -1)
        condition = wmo_codes.get(code, "Code {}".format(code))

        return (
            "Weather in {}: {}, Temperature: {}C, Wind: {} km/h, Humidity: {}"
        ).format(display_name, condition, cw.get("temperature"), cw.get("windspeed"), humidity)

    def get_declaration(self):
        return {
            "name": self.name,
            "description": "Get the current weather conditions for a given city.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "city": {
                        "type": "STRING",
                        "description": "Name of the city, e.g. 'Paris' or 'New York'.",
                    }
                },
                "required": ["city"],
            },
        }


# ── 4. Currency Converter ─────────────────────────────────────────────────────

class CurrencyTool(BaseTool):
    """Convert an amount between currencies using frankfurter.app (free, no key)."""

    _API_URL = "https://api.frankfurter.app/latest"

    @property
    def name(self):
        return "convert_currency"

    def execute(self, amount=1.0, from_currency="USD", to_currency="EUR"):
        try:
            resp = requests.get(
                self._API_URL,
                params={
                    "amount": amount,
                    "from": from_currency.upper(),
                    "to": to_currency.upper(),
                },
                timeout=8,
            ).json()
        except requests.RequestException as exc:
            return "Error: could not reach currency service: {}".format(exc)

        if "error" in resp:
            return "Error: {}".format(resp["error"])

        converted = resp.get("rates", {}).get(to_currency.upper())
        if converted is None:
            return "Error: unsupported currency '{}'.".format(to_currency)

        return "{} {} = {} {} (rate as of {})".format(
            amount, from_currency.upper(), converted,
            to_currency.upper(), resp.get("date", "unknown"))

    def get_declaration(self):
        return {
            "name": self.name,
            "description": "Convert a monetary amount from one currency to another using live exchange rates.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "amount": {
                        "type": "NUMBER",
                        "description": "The amount to convert.",
                    },
                    "from_currency": {
                        "type": "STRING",
                        "description": "ISO 4217 source currency code, e.g. 'USD'.",
                    },
                    "to_currency": {
                        "type": "STRING",
                        "description": "ISO 4217 target currency code, e.g. 'EUR'.",
                    },
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        }


# ── 5. Translation  (Custom tool #1) ──────────────────────────────────────────

class TranslationTool(BaseTool):
    """
    Translate text using the free MyMemory translation API.
    Custom tool #1 — no API key required.
    """

    _API_URL = "https://api.mymemory.translated.net/get"

    @property
    def name(self):
        return "translate_text"

    def execute(self, text="", source_language="en", target_language="es"):
        if not text:
            return "Error: text is required."
        try:
            resp = requests.get(
                self._API_URL,
                params={"q": text, "langpair": "{}|{}".format(source_language, target_language)},
                timeout=8,
            ).json()
        except requests.RequestException as exc:
            return "Error: could not reach translation service: {}".format(exc)

        status = resp.get("responseStatus")
        if status != 200:
            return "Error: translation failed (status {}): {}".format(
                status, resp.get("responseDetails", ""))

        translated = resp.get("responseData", {}).get("translatedText", "")
        if not translated:
            return "Error: empty translation returned."

        return 'Translation ({} -> {}): "{}"'.format(
            source_language, target_language, translated)

    def get_declaration(self):
        return {
            "name": self.name,
            "description": (
                "Translate a piece of text from one language to another. "
                "Use ISO 639-1 two-letter language codes (e.g. 'en', 'fr', 'de', 'ja', 'ar')."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "text": {
                        "type": "STRING",
                        "description": "The text to translate.",
                    },
                    "source_language": {
                        "type": "STRING",
                        "description": "ISO 639-1 source language code, e.g. 'en'.",
                    },
                    "target_language": {
                        "type": "STRING",
                        "description": "ISO 639-1 target language code, e.g. 'fr'.",
                    },
                },
                "required": ["text", "target_language"],
            },
        }


# ── 6. File Reader  (Custom tool #2) ──────────────────────────────────────────

class FileReaderTool(BaseTool):
    """
    Read the contents of a local text file and return them to the agent.
    Custom tool #2.
    """

    MAX_CHARS = 4000

    @property
    def name(self):
        return "read_local_file"

    def execute(self, file_path=""):
        if not file_path:
            return "Error: file_path is required."
        path = os.path.expanduser(file_path)
        if not os.path.exists(path):
            return "Error: file '{}' does not exist.".format(path)
        if not os.path.isfile(path):
            return "Error: '{}' is not a regular file.".format(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(self.MAX_CHARS)
            truncated = os.path.getsize(path) > self.MAX_CHARS
            suffix = "\n... [truncated at {} characters]".format(self.MAX_CHARS) if truncated else ""
            return "Contents of '{}':\n{}{}".format(path, content, suffix)
        except PermissionError:
            return "Error: permission denied reading '{}'.".format(path)
        except Exception as exc:
            return "Error reading '{}': {}".format(path, exc)

    def get_declaration(self):
        return {
            "name": self.name,
            "description": (
                "Read the text contents of a local file and return them. "
                "Useful for summarising or answering questions about files on disk."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "file_path": {
                        "type": "STRING",
                        "description": "Absolute or relative path to the file, e.g. 'C:/Users/notes.txt'.",
                    }
                },
                "required": ["file_path"],
            },
        }


# ==============================================================================
# AGENT  (ReAct loop — Reason -> Act -> Observe)
# ==============================================================================

class Agent:
    """
    Orchestrates the ReAct (Reason -> Act -> Observe) loop.

    Responsibilities:
      - Maintain references to MemoryManager, ToolRegistry, and observer(s).
      - Send requests to Gemini with tool declarations.
      - Detect function_call parts and dispatch to ToolRegistry (Strategy).
      - Feed tool results back to Gemini as function_response parts.
      - Repeat until the model returns a plain text final answer.

    The Agent is open for extension (new tools, new observers) without
    modification — it never references concrete tool classes directly (OCP).
    """

    MAX_REACT_STEPS = 10

    def __init__(self, api_key, model_name, registry, memory, observers=None):
        self._model_name = model_name
        self._registry   = registry
        self._memory     = memory
        self._observers  = observers or []
        self._client     = genai.Client(api_key=api_key)
        self._tool_spec  = registry.build_gemini_tool()
        self._config     = types.GenerateContentConfig(
            tools=[self._tool_spec],
            system_instruction=(
                "You are a helpful personal assistant. "
                "Use the provided tools whenever you need real-time or computed information. "
                "Always reason step by step before calling a tool. "
                "When you have all the information you need, give a clear, concise answer."
            ),
        )

    # ── Observer helpers ──────────────────────────────────────────────────────

    def _notify_tool_call(self, name, args):
        for obs in self._observers:
            try:
                obs.on_tool_call(name, args)
            except Exception:
                pass

    def _notify_tool_result(self, name, result):
        for obs in self._observers:
            try:
                obs.on_tool_result(name, result)
            except Exception:
                pass

    def _notify_error(self, ctx, err):
        for obs in self._observers:
            try:
                obs.on_error(ctx, err)
            except Exception:
                pass

    # ── Main entry point ──────────────────────────────────────────────────────

    def chat(self, user_input):
        """
        Process one user turn and return the agent's final text response.

        ReAct loop:
          1. Reason  — send history + user input to Gemini
          2. Act     — if model returns function_call, execute via registry
          3. Observe — append function_response, repeat from step 1
          4. Answer  — return plain text once model stops calling tools
        """
        self._memory.add_user_text(user_input)
        contents = self._memory.get_history()

        for _ in range(self.MAX_REACT_STEPS):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=self._config,
                )
            except Exception as exc:
                self._notify_error("generate_content", exc)
                error_msg = "I'm sorry, I encountered an API error: {}".format(exc)
                self._memory.add_model_text(error_msg)
                return error_msg

            candidate  = response.candidates[0]
            parts      = candidate.content.parts

            # Separate function calls from text parts
            function_calls = [p for p in parts if p.function_call and p.function_call.name]
            text_parts     = [p for p in parts if p.text]

            # ── No function calls → final answer ──────────────────────────────
            if not function_calls:
                final_text = " ".join(p.text for p in text_parts).strip()
                if not final_text:
                    final_text = "I wasn't able to produce a response. Please try again."
                self._memory.add_model_text(final_text)
                return final_text

            # ── Append the model's full turn (text + calls) ───────────────────
            contents.append(candidate.content)

            # ── Execute each function call (Act -> Observe) ───────────────────
            tool_response_parts = []
            for fc_part in function_calls:
                fc        = fc_part.function_call
                tool_name = fc.name
                args      = dict(fc.args) if fc.args else {}

                self._notify_tool_call(tool_name, args)
                result = self._registry.execute(tool_name, args)
                self._notify_tool_result(tool_name, result)

                tool_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tool_name,
                            response={"result": result},
                        )
                    )
                )

            # Append tool results as a "user" turn so the model can observe them
            contents.append(types.Content(role="user", parts=tool_response_parts))

        fallback = "I exceeded the maximum reasoning steps. Please rephrase your question."
        self._memory.add_model_text(fallback)
        return fallback


# ==============================================================================
# BOOTSTRAP — wire everything together
# ==============================================================================

def build_agent():
    """Instantiate and wire all components (composition root)."""

    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyAfi97-eRA9r4Ma4yCIsFBFh2SIs3n2CVk")

    registry = ToolRegistry()
    for tool in [
        CalculatorTool(),
        TimeTool(),
        WeatherTool(),
        CurrencyTool(),
        TranslationTool(),
        FileReaderTool(),
    ]:
        registry.register(tool)

    return Agent(
        api_key=api_key,
        model_name="gemini-2.5-flash",
        registry=registry,
        memory=MemoryManager(),
        observers=[ConsoleLogger()],
    )


# ==============================================================================
# CLI
# ==============================================================================

_BANNER = """
+----------------------------------------------------------+
|          Adaptive AI Agent  -  Design Patterns           |
|  Tools: calculator, time, weather, currency,             |
|         translate, read_file                             |
|  Commands: 'quit'/'exit' to stop, 'clear' to reset       |
+----------------------------------------------------------+
"""

_BLUE  = "\033[94m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"


def main():
    print(_BANNER)

    try:
        agent = build_agent()
    except Exception as exc:
        print("Fatal error during initialisation: {}".format(exc))
        traceback.print_exc()
        return

    print("{}Agent ready. Type your message below.{}\n".format(_BOLD, _RESET))

    while True:
        try:
            user_input = input("{}You:{} ".format(_BLUE, _RESET)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            agent._memory.clear()
            print("Conversation history cleared.\n")
            continue

        print()
        response = agent.chat(user_input)
        print("{}Agent:{} {}\n".format(_BOLD, _RESET, response))


if __name__ == "__main__":
    main()
