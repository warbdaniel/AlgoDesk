"""
Autonomous trading agent powered by the Claude API.

Uses the Anthropic Python SDK with tool use to let Claude interact with
all AlgoDesk services (regime-detector, fix-api, data-pipeline, dashboard)
through a controlled agentic loop.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

import trading_tools as tools

logger = logging.getLogger("claude_agent")

# ── Tool catalogue ───────────────────────────────────────────────
# Each entry maps a tool name to (callable, description, JSON-schema properties, required).

@dataclass
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    fn: Any  # callable(service_urls, **input) -> str
    read_only: bool = True  # False = can modify state (orders, kill switch)


def _build_tool_specs() -> list[ToolSpec]:
    """Return the full catalogue of trading tools."""
    return [
        # ── regime detector ──────────────────────────────────────
        ToolSpec(
            name="detect_regime",
            description=(
                "Classify the current market regime (STRONG_TREND, MILD_TREND, "
                "RANGING, CHOPPY) for a symbol on a given timeframe."
            ),
            properties={
                "symbol": {"type": "string", "description": "cTrader symbol ID, e.g. '1' for EURUSD"},
                "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], "description": "Candle timeframe"},
            },
            required=["symbol"],
            fn=lambda urls, **kw: tools.detect_regime(urls["regime_detector"], **kw),
        ),
        ToolSpec(
            name="regime_health",
            description="Check regime-detector service health.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.regime_health(urls["regime_detector"]),
        ),

        # ── fix-api  prices ──────────────────────────────────────
        ToolSpec(
            name="get_prices",
            description="Get latest bid/ask prices for one or all subscribed symbols.",
            properties={
                "symbol": {"type": "string", "description": "Symbol ID. Omit for all symbols."},
            },
            required=[],
            fn=lambda urls, **kw: tools.get_prices(urls["fix_api"], kw.get("symbol")),
        ),
        ToolSpec(
            name="get_candles",
            description="Get recent 1-minute candles from the live FIX feed.",
            properties={
                "symbol": {"type": "string", "description": "Symbol ID"},
                "count": {"type": "integer", "description": "Number of candles (default 50)"},
            },
            required=["symbol"],
            fn=lambda urls, **kw: tools.get_candles(urls["fix_api"], kw["symbol"], kw.get("count", 50)),
        ),
        ToolSpec(
            name="subscribe_symbol",
            description="Subscribe to live price updates for a symbol.",
            properties={"symbol": {"type": "string", "description": "Symbol ID"}},
            required=["symbol"],
            fn=lambda urls, **kw: tools.subscribe_symbol(urls["fix_api"], kw["symbol"]),
            read_only=False,
        ),
        ToolSpec(
            name="unsubscribe_symbol",
            description="Unsubscribe from live price updates.",
            properties={"symbol": {"type": "string", "description": "Symbol ID"}},
            required=["symbol"],
            fn=lambda urls, **kw: tools.unsubscribe_symbol(urls["fix_api"], kw["symbol"]),
            read_only=False,
        ),

        # ── fix-api  orders ──────────────────────────────────────
        ToolSpec(
            name="place_order",
            description=(
                "Submit an order through the FIX trade connection. "
                "Supports MARKET, LIMIT, and STOP order types."
            ),
            properties={
                "symbol": {"type": "string", "description": "cTrader symbol ID"},
                "side": {"type": "string", "enum": ["BUY", "SELL"], "description": "Order side"},
                "quantity": {"type": "number", "description": "Lot size (e.g. 0.01)"},
                "order_type": {"type": "string", "enum": ["MARKET", "LIMIT", "STOP"], "description": "Order type (default MARKET)"},
                "price": {"type": "number", "description": "Limit price (for LIMIT orders)"},
                "stop_price": {"type": "number", "description": "Stop price (for STOP orders)"},
                "time_in_force": {"type": "string", "enum": ["GTC", "IOC", "FOK"], "description": "Time in force (default GTC)"},
            },
            required=["symbol", "side", "quantity"],
            fn=lambda urls, **kw: tools.place_order(
                urls["fix_api"],
                symbol=kw["symbol"],
                side=kw["side"],
                quantity=kw["quantity"],
                order_type=kw.get("order_type", "MARKET"),
                price=kw.get("price", 0.0),
                stop_price=kw.get("stop_price", 0.0),
                time_in_force=kw.get("time_in_force", "GTC"),
            ),
            read_only=False,
        ),
        ToolSpec(
            name="cancel_order",
            description="Cancel a pending order by its client order ID.",
            properties={"cl_ord_id": {"type": "string", "description": "Client order ID"}},
            required=["cl_ord_id"],
            fn=lambda urls, **kw: tools.cancel_order(urls["fix_api"], kw["cl_ord_id"]),
            read_only=False,
        ),
        ToolSpec(
            name="get_orders",
            description="List all open/pending orders.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_orders(urls["fix_api"]),
        ),
        ToolSpec(
            name="get_positions",
            description="List current open positions with unrealized P&L.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_positions(urls["fix_api"]),
        ),
        ToolSpec(
            name="get_account",
            description="Get account balance, equity, and margin info.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_account(urls["fix_api"]),
        ),

        # ── fix-api  risk ────────────────────────────────────────
        ToolSpec(
            name="get_risk_status",
            description="Get current risk manager status and utilisation.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_risk_status(urls["fix_api"]),
        ),
        ToolSpec(
            name="get_risk_violations",
            description="List recent risk rule violations.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_risk_violations(urls["fix_api"]),
        ),
        ToolSpec(
            name="activate_kill_switch",
            description="EMERGENCY: activate kill switch to halt all trading.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.activate_kill_switch(urls["fix_api"]),
            read_only=False,
        ),
        ToolSpec(
            name="deactivate_kill_switch",
            description="Deactivate kill switch to resume trading.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.deactivate_kill_switch(urls["fix_api"]),
            read_only=False,
        ),
        ToolSpec(
            name="fix_health",
            description="Check FIX API service health and connection status.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.fix_health(urls["fix_api"]),
        ),
        ToolSpec(
            name="connection_health",
            description="Get detailed FIX connection diagnostics.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.connection_health(urls["fix_api"]),
        ),

        # ── data pipeline ────────────────────────────────────────
        ToolSpec(
            name="get_features",
            description=(
                "Compute technical indicator feature vector (42+ indicators: "
                "SMA, EMA, MACD, RSI, ATR, Bollinger, Stochastic, ADX, etc.) "
                "for a symbol on a given interval."
            ),
            properties={
                "symbol": {"type": "string", "description": "Symbol ID"},
                "interval": {"type": "string", "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], "description": "Candle interval"},
            },
            required=["symbol"],
            fn=lambda urls, **kw: tools.get_features(urls["data_pipeline"], kw["symbol"], kw.get("interval", "15m")),
        ),
        ToolSpec(
            name="get_feature_names",
            description="List all feature names in the ML vector.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_feature_names(urls["data_pipeline"]),
        ),
        ToolSpec(
            name="get_pipeline_candles",
            description="Get stored OHLCV candles from the data pipeline database.",
            properties={
                "symbol": {"type": "string", "description": "Symbol ID"},
                "interval": {"type": "string", "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], "description": "Candle interval"},
                "limit": {"type": "integer", "description": "Number of candles (default 50)"},
            },
            required=["symbol"],
            fn=lambda urls, **kw: tools.get_pipeline_candles(
                urls["data_pipeline"], kw["symbol"], kw.get("interval", "15m"), kw.get("limit", 50)
            ),
        ),
        ToolSpec(
            name="get_latest_tick",
            description="Get the most recent tick (bid/ask) for a symbol.",
            properties={"symbol": {"type": "string", "description": "Symbol ID"}},
            required=["symbol"],
            fn=lambda urls, **kw: tools.get_latest_tick(urls["data_pipeline"], kw["symbol"]),
        ),
        ToolSpec(
            name="get_symbols",
            description="List all registered symbols in the data pipeline.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_symbols(urls["data_pipeline"]),
        ),
        ToolSpec(
            name="get_pipeline_stats",
            description="Get data pipeline statistics (tick counts, candle counts, etc.).",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_pipeline_stats(urls["data_pipeline"]),
        ),
        ToolSpec(
            name="get_events",
            description="Get recent events from the data bus (ticks, regime changes, signals, fills).",
            properties={
                "event_type": {"type": "string", "enum": ["tick", "candle_close", "regime_change", "signal", "order_fill", "risk_alert", "system"], "description": "Filter by event type (optional)"},
                "limit": {"type": "integer", "description": "Number of events (default 50)"},
            },
            required=[],
            fn=lambda urls, **kw: tools.get_events(urls["data_pipeline"], kw.get("event_type"), kw.get("limit", 50)),
        ),
        ToolSpec(
            name="pipeline_health",
            description="Check data pipeline service health.",
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.pipeline_health(urls["data_pipeline"]),
        ),

        # ── dashboard ────────────────────────────────────────────
        ToolSpec(
            name="get_trades",
            description="Query trade history with optional symbol/regime filters.",
            properties={
                "symbol": {"type": "string", "description": "Filter by symbol (optional)"},
                "regime": {"type": "string", "description": "Filter by regime (optional)"},
                "limit": {"type": "integer", "description": "Number of trades (default 50)"},
            },
            required=[],
            fn=lambda urls, **kw: tools.get_trades(
                urls["dashboard"], kw.get("symbol"), kw.get("regime"), kw.get("limit", 50)
            ),
        ),
        ToolSpec(
            name="get_analytics",
            description=(
                "Retrieve computed performance metrics: Sharpe ratio, profit factor, "
                "win rate, expected value, max drawdown, avg trade duration."
            ),
            properties={},
            required=[],
            fn=lambda urls, **kw: tools.get_analytics(urls["dashboard"]),
        ),
    ]


SYSTEM_PROMPT = """\
You are the AlgoDesk Trading Agent — an autonomous assistant integrated into \
a live FX algorithmic trading infrastructure. You have direct access to:

• **Regime Detector** – classifies market conditions (STRONG_TREND, MILD_TREND, RANGING, CHOPPY)
• **FIX API** – live bid/ask prices, order placement/cancellation, positions, account info
• **Data Pipeline** – stored candles, 42+ technical indicators, event bus
• **Dashboard** – historical trade log, performance analytics (Sharpe, PF, win rate, drawdown)

**Guidelines**
1. Always check the current regime and key indicators before recommending or placing trades.
2. Respect risk limits. Check risk status before placing orders. Never bypass the kill switch.
3. When presenting analysis, cite the specific indicator values and regime classification.
4. For order placement: state the full order parameters (symbol, side, qty, type) and \
   explain the rationale before executing.
5. If a service is down, report it clearly and suggest what can still be done.
6. Prefer structured, concise answers. Use tables for multi-symbol comparisons.
7. You are allowed to call multiple tools in sequence to build a complete picture.
"""


# ── Agent core ───────────────────────────────────────────────────

@dataclass
class AgentConfig:
    model: str = "claude-opus-4-6"
    max_tokens: int = 16000
    max_tool_rounds: int = 20
    read_only: bool = False
    system_prompt: str = SYSTEM_PROMPT


@dataclass
class AgentResult:
    response_text: str
    tool_calls_made: list[dict] = field(default_factory=list)
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_sec: float = 0.0


class TradingAgent:
    """Stateless Claude-powered trading agent with tool use."""

    def __init__(self, config: AgentConfig, service_urls: dict[str, str]):
        self.config = config
        self.service_urls = service_urls
        self.client = anthropic.Anthropic()

        all_specs = _build_tool_specs()
        if config.read_only:
            all_specs = [s for s in all_specs if s.read_only]
        self._specs_by_name: dict[str, ToolSpec] = {s.name: s for s in all_specs}

    # -- public API ---------------------------------------------------

    def query(self, user_message: str, conversation: list[dict] | None = None) -> AgentResult:
        """
        Run the agentic loop: send user message, execute tools Claude
        requests, feed results back, repeat until Claude stops.
        """
        messages: list[dict] = list(conversation) if conversation else []
        messages.append({"role": "user", "content": user_message})

        tool_defs = self._build_tool_defs()
        tool_log: list[dict] = []
        total_in = 0
        total_out = 0
        t0 = time.time()

        for round_num in range(1, self.config.max_tool_rounds + 1):
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                thinking={"type": "adaptive"},
                tools=tool_defs,
                messages=messages,
            )

            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens

            # Append assistant response to conversation
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text = self._extract_text(response)
                return AgentResult(
                    response_text=text,
                    tool_calls_made=tool_log,
                    rounds=round_num,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    elapsed_sec=time.time() - t0,
                )

            if response.stop_reason != "tool_use":
                # Unexpected stop reason — return what we have
                text = self._extract_text(response)
                return AgentResult(
                    response_text=text or f"(stopped: {response.stop_reason})",
                    tool_calls_made=tool_log,
                    rounds=round_num,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    elapsed_sec=time.time() - t0,
                )

            # Execute every tool_use block and collect results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_text = self._execute_tool(block.name, block.input)
                tool_log.append({
                    "round": round_num,
                    "tool": block.name,
                    "input": block.input,
                    "output_preview": result_text[:200],
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        # Exhausted rounds
        return AgentResult(
            response_text="(agent reached maximum tool rounds without finishing)",
            tool_calls_made=tool_log,
            rounds=self.config.max_tool_rounds,
            input_tokens=total_in,
            output_tokens=total_out,
            elapsed_sec=time.time() - t0,
        )

    def query_streaming(self, user_message: str, conversation: list[dict] | None = None):
        """
        Generator that yields (event_type, data) tuples for real-time
        streaming to the client. Event types:
          'text'        – partial text delta
          'tool_call'   – tool being executed
          'tool_result' – tool result summary
          'done'        – final AgentResult
        """
        messages: list[dict] = list(conversation) if conversation else []
        messages.append({"role": "user", "content": user_message})

        tool_defs = self._build_tool_defs()
        tool_log: list[dict] = []
        total_in = 0
        total_out = 0
        t0 = time.time()

        for round_num in range(1, self.config.max_tool_rounds + 1):
            collected_text = []

            with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                thinking={"type": "adaptive"},
                tools=tool_defs,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            collected_text.append(event.delta.text)
                            yield ("text", event.delta.text)

                response = stream.get_final_message()

            total_in += response.usage.input_tokens
            total_out += response.usage.output_tokens
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                yield ("done", AgentResult(
                    response_text="".join(collected_text),
                    tool_calls_made=tool_log,
                    rounds=round_num,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    elapsed_sec=time.time() - t0,
                ))
                return

            if response.stop_reason != "tool_use":
                yield ("done", AgentResult(
                    response_text="".join(collected_text) or f"(stopped: {response.stop_reason})",
                    tool_calls_made=tool_log,
                    rounds=round_num,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    elapsed_sec=time.time() - t0,
                ))
                return

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                yield ("tool_call", {"tool": block.name, "input": block.input})
                result_text = self._execute_tool(block.name, block.input)
                tool_log.append({
                    "round": round_num,
                    "tool": block.name,
                    "input": block.input,
                    "output_preview": result_text[:200],
                })
                yield ("tool_result", {"tool": block.name, "preview": result_text[:300]})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})

        yield ("done", AgentResult(
            response_text="(agent reached maximum tool rounds without finishing)",
            tool_calls_made=tool_log,
            rounds=self.config.max_tool_rounds,
            input_tokens=total_in,
            output_tokens=total_out,
            elapsed_sec=time.time() - t0,
        ))

    # -- internal -----------------------------------------------------

    def _build_tool_defs(self) -> list[dict]:
        """Convert ToolSpecs into Claude API tool definitions."""
        defs = []
        for spec in self._specs_by_name.values():
            tool_def: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "input_schema": {
                    "type": "object",
                    "properties": spec.properties,
                    "required": spec.required,
                },
            }
            defs.append(tool_def)
        return defs

    def _execute_tool(self, name: str, tool_input: dict) -> str:
        """Dispatch a tool call and return the result as a string."""
        spec = self._specs_by_name.get(name)
        if not spec:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            return spec.fn(self.service_urls, **tool_input)
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            return json.dumps({"error": str(exc)})

    @staticmethod
    def _extract_text(response) -> str:
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)
