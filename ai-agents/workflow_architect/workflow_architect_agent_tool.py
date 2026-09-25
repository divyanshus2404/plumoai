from __future__ import annotations

"""
Workflow Architect Agent (app_code: workflow_architect)

Designs a complete PlumoAI workflow from a plain-language spec. The user
describes a goal, the inputs they will provide, and the outputs they want; the
agent asks the LLM to produce an ordered, node-by-node build plan and then
*deterministically validates* every node in that plan against the known node
catalog, so the plan never references a node that does not exist.

No credentials, no side effects: prompt in, validated build plan out. It does
not create the workflow on the canvas (that requires the platform's internal
workflow-graph API) — it produces the design a user (or a future create-API)
can build from.

Self-contained plugin: all logic lives in this folder.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from backend.services.ai_agents.base_tool_agent import BaseToolAgent

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 1600
DEFAULT_TEMPERATURE = 0.3  # design work wants low variance


class AgentEvent:
    THOUGHT = "thought"
    RESULT = "result"
    FINAL = "final"
    ERROR = "error"


def event(event_type: str, content: Any) -> Dict:
    return {
        "id": str(uuid.uuid4()),
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "content": content,
    }


# ---------------------------------------------------------------------------
# Tool catalog — the tools actually preloaded in this PlumoAI setup (built-in
# workflow nodes + the installed AI Agent plugins). Each entry is
# (category, name, what it is for, needs a connected account). This is the
# single source of truth: the design prompt is grounded on it (so the agent
# only picks appropriate, real tools), the break-checker validates against it,
# and the "tools reference" is rendered from it. Callers can add tools at
# runtime via tool_args["extra_nodes"] / app_config["extra_nodes"].
# ---------------------------------------------------------------------------
TOOL_CATALOG: List[Tuple[str, str, str, bool]] = [
    # category, name, purpose, needs_account
    ("Triggers", "Schedule Trigger", "start on a one-time or recurring schedule", False),
    ("Triggers", "Webhook Trigger", "start when an external webhook is called", False),
    ("Triggers", "Manual Trigger", "start manually when you run it", False),
    ("Triggers", "Execute Sub-workflow Trigger", "start when another workflow calls this one", False),

    ("Logic & Flow", "IF Else", "split the path on one true/false condition", False),
    ("Logic & Flow", "Switch", "route down one of several branches by rules", False),
    ("Logic & Flow", "Wait", "pause the workflow (a delay, or until a time)", False),
    ("Logic & Flow", "Set Variables", "set or rename fields for later steps to use", False),
    ("Logic & Flow", "Split In Batches", "loop over items in fixed-size batches", False),
    ("Logic & Flow", "Execute Sub-workflow", "run another workflow as a step", False),
    ("Logic & Flow", "End Workflow", "stop the workflow cleanly", False),
    ("Logic & Flow", "Loop Executor", "expand a batch goal into per-item steps", False),

    ("Data", "Filter", "keep only items matching conditions", False),
    ("Data", "Limit", "keep the first or last N items", False),
    ("Data", "Remove Duplicates", "drop repeated items", False),
    ("Data", "Split Out", "turn a list into one item each", False),
    ("Data", "Sort", "order items by a field", False),
    ("Data", "JavaScript", "run custom JavaScript to transform or filter data", False),

    ("AI & Reasoning", "AI Agent", "one LLM call — free-form reasoning or answer", False),
    ("AI & Reasoning", "AI Writer", "draft emails, messages, reports, documents", False),
    ("AI & Reasoning", "Web Search", "search the web for current info and facts", False),
    ("AI & Reasoning", "Knowledge Base Search", "search the company / employee knowledge base", False),
    ("AI & Reasoning", "Chart Maker", "build charts from structured data", False),
    ("AI & Reasoning", "Image Generation", "generate images from a text description", False),
    ("AI & Reasoning", "Calculator & DateTime", "do math and date / time calculations", False),
    ("AI & Reasoning", "Memory", "store and recall long-term user facts", False),

    ("Communication", "Gmail", "send and read email via Gmail", True),
    ("Communication", "SendGrid", "send transactional email via SendGrid", True),
    ("Communication", "Discord", "post and manage Discord messages and channels", True),
    ("Communication", "WhatsApp Business", "send WhatsApp Business messages", True),
    ("Communication", "Google Chat", "post and manage Google Chat messages", True),
    ("Communication", "Call", "place and manage voice calls", True),
    ("Communication", "LinkedIn", "post and manage LinkedIn content", True),
    ("Communication", "YouTube", "search and manage YouTube videos and channels", True),

    ("Scheduling & Meetings", "Cal.com", "manage Cal.com scheduling and bookings", True),
    ("Scheduling & Meetings", "Google Calendar", "manage calendar events and availability", True),
    ("Scheduling & Meetings", "Google Meet", "manage Meet spaces, recordings, transcripts", True),

    ("Data & CRM", "Google Sheets", "read and write Google Sheets", True),
    ("Data & CRM", "Google Drive", "manage Google Drive files and folders", True),
    ("Data & CRM", "Apollo.io", "pull B2B contacts and accounts from Apollo", True),
    ("Data & CRM", "GetLeads.io", "B2B contact database (402M+ contacts)", True),
    ("Data & CRM", "Apify", "run web-scraping actors on Apify", True),
    ("Data & CRM", "SQL Server", "query Microsoft SQL Server data", True),
    ("Data & CRM", "Notion", "read and write Notion workspace content", True),
    ("Data & CRM", "Upwork", "work with Upwork on the connected account", True),
    ("Data & CRM", "PlumoAI", "access PlumoAI workspaces, AI Employees, tables", False),
]

# Category order for rendering the tools reference.
_CATEGORY_ORDER = [
    "Triggers", "Logic & Flow", "Data", "Code", "AI & Reasoning",
    "Communication", "Scheduling & Meetings", "Data & CRM",
]

_ALL_TOOL_NAMES: List[str] = [name for _, name, _, _ in TOOL_CATALOG]
_PURPOSE: Dict[str, str] = {name: purpose for _, name, purpose, _ in TOOL_CATALOG}


def _norm(name: str) -> str:
    """Normalize a node name for tolerant matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


# Common aliases the LLM (or a user) might use for a real node.
_ALIASES: Dict[str, str] = {
    _norm("if"): "IF Else",
    _norm("if/else"): "IF Else",
    _norm("if else"): "IF Else",
    _norm("schedule"): "Schedule Trigger",
    _norm("cron"): "Schedule Trigger",
    _norm("webhook"): "Webhook Trigger",
    _norm("manual"): "Manual Trigger",
    _norm("set"): "Set Variables",
    _norm("set variable"): "Set Variables",
    _norm("edit fields"): "Set Variables",
    _norm("loop"): "Split In Batches",
    _norm("split in batch"): "Split In Batches",
    _norm("dedupe"): "Remove Duplicates",
    _norm("deduplicate"): "Remove Duplicates",
    _norm("code"): "JavaScript",
    _norm("js"): "JavaScript",
    _norm("websearch"): "Web Search",
    _norm("google sheet"): "Google Sheets",
    _norm("sheets"): "Google Sheets",
    _norm("end"): "End Workflow",
}

# Tools that call an external account and need a connected credential before the
# workflow can run. Derived from the catalog — flagged as warnings (not breaks)
# by the checker: the workflow is valid, but the account must be connected first.
_NEEDS_CREDENTIAL = frozenset(name for _, name, _, cred in TOOL_CATALOG if cred)

# Trigger tools — a workflow needs exactly one, as its first node.
_TRIGGERS = frozenset(name for cat, name, _, _ in TOOL_CATALOG if cat == "Triggers")


def _build_index(extra_nodes: Optional[List[str]]) -> Dict[str, str]:
    """Map normalized name -> canonical display name, across the catalog + aliases + extras."""
    index: Dict[str, str] = {}
    for name in _ALL_TOOL_NAMES:
        index[_norm(name)] = name
    index.update(_ALIASES)
    for n in extra_nodes or []:
        if isinstance(n, str) and n.strip():
            index[_norm(n)] = n.strip()
    return index


def _grouped_catalog() -> "Dict[str, List[Tuple[str, str, bool]]]":
    """category -> [(name, purpose, needs_account)], in _CATEGORY_ORDER."""
    grouped: Dict[str, List[Tuple[str, str, bool]]] = {c: [] for c in _CATEGORY_ORDER}
    for cat, name, purpose, cred in TOOL_CATALOG:
        grouped.setdefault(cat, []).append((name, purpose, cred))
    return grouped


def _catalog_text() -> str:
    """Catalog for the design prompt: each tool as `name — purpose` so the model
    can choose tools that actually fit the task."""
    lines: List[str] = []
    for cat, items in _grouped_catalog().items():
        if not items:
            continue
        lines.append(f"{cat}:")
        for name, purpose, _ in items:
            lines.append(f"  - {name} — {purpose}")
    return "\n".join(lines)


def _tools_reference() -> str:
    """Human-readable tools reference for the chat: what each tool is for and
    whether it needs a connected account (🔑)."""
    lines: List[str] = ["# 🧰 Available tools", "", "_What each tool is for — 🔑 means it needs a connected account._", ""]
    for cat, items in _grouped_catalog().items():
        if not items:
            continue
        lines.append(f"## {cat}")
        for name, purpose, cred in items:
            key = " 🔑" if cred else ""
            lines.append(f"- **{name}**{key} — {purpose}")
        lines.append("")
    return "\n".join(lines).strip()


def _wants_tools_reference(text: str) -> bool:
    """True when the user is asking to see the tool list rather than design a
    workflow — e.g. 'what tools are available', 'which tool for X', 'list tools'."""
    t = (text or "").lower()
    if len(t) > 140:  # a long spec is a design request, not a catalog question
        return False
    phrases = (
        "what tools", "which tools", "list tools", "available tools", "tools available",
        "what tool", "which tool", "show tools", "show me the tools", "what can you use",
        "what tools do", "tools reference", "list of tools", "what's available",
    )
    return any(p in t for p in phrases)


class WorkflowArchitectAgentTool(BaseToolAgent):
    """
    Workflow Architect: turns a plain-language spec into a validated,
    node-by-node workflow build plan.
    """

    TOOL_NAME = "Workflow Architect"
    APP_CODE = "workflow_architect"

    TOOL_DESCRIPTION = """Workflow Architect: designs a complete PlumoAI workflow from a plain-language description, so a user does not have to design it node-by-node by hand.

USER PROVIDES:
- Goal (what the workflow should achieve)
- Inputs (what data/values will be provided or triggered)
- Outputs (what the workflow should produce)
- Optional: schedule/trigger preference, any specific apps to use

OUTPUT:
- An ordered, node-by-node build plan: each node, its category, its configuration, and its purpose
- The connection order (which node feeds which)
- A validation report flagging any node not found in the available catalog

USE WHEN:
- User asks to "design", "plan", "build", or "architect" a workflow / automation
- User describes an automation in words and wants to know which nodes to add and how to wire them
- User asks "how would I build a workflow that ..."
- User asks what tools are available or "which tool is for what" (returns a tools reference)

NOTE: This produces the design/plan. It does not draw the workflow on the canvas automatically.
"""

    def __init__(
        self,
        llm_provider: Any,
        agent_id: Optional[str] = None,
        token: Optional[str] = None,
        company_id: Optional[str] = None,
        user_id: Optional[int] = None,
        app_config: Optional[Dict[str, Any]] = None,
    ):
        self.llm_provider = llm_provider
        self.agent_id = agent_id or ""
        self.token = token
        self.company_id = company_id
        self.user_id = user_id
        self.app_config = app_config or {}

    @classmethod
    def get_tool_responsibility(cls) -> str:
        return cls.TOOL_DESCRIPTION

    def get_description(self) -> str:
        return self.get_tool_responsibility()

    # ---- input parsing -----------------------------------------------------

    def _extract_params(self, user_query: str, tool_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pull spec/goal/inputs/outputs/extra_nodes from tool_args, JSON, or raw text."""
        args = dict(tool_args or {})
        s = (user_query or "").strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if v is not None:
                            args.setdefault(str(k).strip().lower(), v)
            except json.JSONDecodeError:
                pass

        goal = str(args.get("goal") or args.get("spec") or args.get("prompt") or args.get("description") or "").strip()
        if not goal:
            goal = s if s and not s.startswith("{") else ""

        inputs = args.get("inputs")
        outputs = args.get("outputs")

        extra_nodes = args.get("extra_nodes")
        if isinstance(extra_nodes, str):
            extra_nodes = [p.strip() for p in extra_nodes.split(",") if p.strip()]
        if not isinstance(extra_nodes, list):
            extra_nodes = []
        # allow deployment-level catalog extension too
        cfg_extra = self.app_config.get("extra_nodes")
        if isinstance(cfg_extra, list):
            extra_nodes = extra_nodes + [n for n in cfg_extra if isinstance(n, str)]

        return {
            "goal": goal,
            "inputs": inputs,
            "outputs": outputs,
            "extra_nodes": extra_nodes,
        }

    # ---- LLM prompt --------------------------------------------------------

    def _system_prompt(self) -> str:
        return (
            "You are a PlumoAI Workflow Architect. You design workflows for a node-based "
            "automation editor. A workflow is an ordered chain of nodes; each node does one "
            "thing and passes its output to the next.\n\n"
            "You MUST design using ONLY tools from this catalog (name — what it is for):\n"
            f"{_catalog_text()}\n\n"
            "Rules:\n"
            "1. Start with exactly one trigger node (Schedule/Webhook/Manual/Execute Sub-workflow Trigger).\n"
            "2. Use the real tool names above, spelled exactly.\n"
            "3. Choose each tool by its stated purpose — pick the tool whose 'what it is for' matches the step. Do not use a tool for something it is not meant to do.\n"
            "4. Prefer built-in Data / Logic nodes over JavaScript when one fits.\n"
            "5. Use Split Out before per-item steps; Remove Duplicates/Filter to clean; Switch/IF Else for branching.\n"
            "6. Keep it as simple as the goal allows — no unnecessary tools.\n\n"
            "Respond with ONLY a JSON object, no prose, in this exact shape:\n"
            '{\n'
            '  "workflow_name": "string",\n'
            '  "summary": "one sentence",\n'
            '  "nodes": [\n'
            '    {"step": 1, "node": "Schedule Trigger", "category": "Trigger", '
            '"config": "what to set on this node", "purpose": "why this node is here"}\n'
            '  ],\n'
            '  "connections": "1 -> 2 -> 3",\n'
            '  "notes": "optional caveats or add-ons"\n'
            "}"
        )

    def _user_prompt(self, goal: str, inputs: Any, outputs: Any) -> str:
        parts = [f"GOAL:\n{goal}"]
        if inputs:
            parts.append(f"\nINPUTS THE USER WILL PROVIDE:\n{inputs if isinstance(inputs, str) else json.dumps(inputs)}")
        if outputs:
            parts.append(f"\nOUTPUTS THE USER WANTS:\n{outputs if isinstance(outputs, str) else json.dumps(outputs)}")
        parts.append("\nDesign the workflow now. JSON only.")
        return "\n".join(parts)

    # ---- output handling ---------------------------------------------------

    def _parse_json_loose(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        raw = text.strip()
        # strip markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass
        first = raw.find("{")
        last = raw.rfind("}")
        if first < 0 or last <= first:
            return None
        try:
            obj = json.loads(raw[first : last + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _validate(self, plan: Dict[str, Any], extra_nodes: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Check every node in the plan against the catalog. Returns (warnings, normalized_nodes)."""
        index = _build_index(extra_nodes)
        warnings: List[str] = []
        norm_nodes: List[Dict[str, Any]] = []

        nodes = plan.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            warnings.append("Plan contained no nodes.")
            return warnings, norm_nodes

        trigger_categories = {_norm("Trigger")}
        has_trigger = False

        for i, n in enumerate(nodes, 1):
            if not isinstance(n, dict):
                warnings.append(f"Step {i}: not a valid node object.")
                continue
            raw_name = str(n.get("node") or "").strip()
            canonical = index.get(_norm(raw_name))
            if canonical is None:
                warnings.append(
                    f"Step {i}: '{raw_name}' is not a known node — replace it with a catalog node."
                )
            else:
                n["node"] = canonical
            if "trigger" in _norm(str(n.get("category") or "")) or "trigger" in _norm(raw_name):
                has_trigger = True
            norm_nodes.append(n)

        if not has_trigger:
            warnings.append("No trigger node found — every workflow should start with a trigger.")

        return warnings, norm_nodes

    def _break_check(self, plan: Dict[str, Any], extra_nodes: List[str]) -> Dict[str, Any]:
        """Structured "is this workflow breaking anywhere?" check. Returns
        {"status": "pass"|"warnings"|"breaks", "checks": [{"level","message"}]}.
        Runs AFTER _validate has normalized node names. Errors ("break") mean the
        workflow cannot run; warnings mean it runs once the user acts (e.g. connects
        an account)."""
        index = _build_index(extra_nodes)
        checks: List[Dict[str, str]] = []

        def add(level: str, message: str) -> None:
            checks.append({"level": level, "message": message})

        nodes = plan.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            add("error", "The workflow has no nodes.")
            return {"status": "breaks", "checks": checks}

        names = [str(n.get("node") or "").strip() for n in nodes if isinstance(n, dict)]

        # 1) exactly one trigger, and it comes first
        trigs = [nm for nm in names if index.get(_norm(nm)) in _TRIGGERS]
        first_is_trigger = bool(names) and index.get(_norm(names[0])) in _TRIGGERS
        if not trigs:
            add("error", "No trigger — the workflow can never start. Add a trigger as the first node.")
        elif len(trigs) > 1:
            add("error", f"{len(trigs)} triggers found — a workflow needs exactly one entry point.")
        elif not first_is_trigger:
            add("error", "The trigger is not the first node — it must come first.")
        else:
            add("ok", "Starts with exactly one trigger.")

        # 2) every node is real
        unknown = [nm for nm in names if index.get(_norm(nm)) is None]
        if unknown:
            for nm in unknown:
                add("error", f"'{nm}' is not a real node — replace it with a catalog node.")
        else:
            add("ok", "Every node exists in the node catalog.")

        # 3) a branch node should not be the last node
        last_canonical = index.get(_norm(names[-1])) if names else None
        if last_canonical in ("Switch", "IF Else"):
            add("warning", f"The last node is '{last_canonical}' — each branch should end in an action or End Workflow.")

        # 4) credential-required nodes
        creds = sorted({index.get(_norm(nm)) for nm in names if index.get(_norm(nm)) in _NEEDS_CREDENTIAL})
        if creds:
            add("warning", "Connect an account before running: " + ", ".join(creds) + ".")

        errs = sum(1 for c in checks if c["level"] == "error")
        warns = sum(1 for c in checks if c["level"] == "warning")
        status = "breaks" if errs else ("warnings" if warns else "pass")
        return {"status": status, "checks": checks}

    @staticmethod
    def _node_emoji(node: str) -> str:
        """A small type glyph for the chat, chosen from the canonical node name."""
        if node in _TRIGGERS:
            return "⏱️"
        if node in ("Switch", "IF Else", "Wait", "Set Variables", "End Workflow",
                    "Execute Sub-workflow", "Split In Batches"):
            return "🔀"
        if node in ("Filter", "Limit", "Remove Duplicates", "Split Out", "Sort"):
            return "🔧"
        if node == "JavaScript":
            return "💻"
        return "🤖"  # AI Agents

    def _render_markdown(self, plan: Dict[str, Any], warnings: List[str], check: Optional[Dict[str, Any]] = None) -> str:
        lines: List[str] = []
        name = str(plan.get("workflow_name") or "Workflow").strip()
        # Title + one-line status badge so the verdict is visible at a glance.
        badge = ""
        if check:
            badge = {
                "pass": "  ✅ ready to build",
                "warnings": "  ⚠️ ready · needs setup",
                "breaks": "  ❌ will not run",
            }.get(check.get("status", ""), "")
        lines.append(f"# {name}{badge}")
        summary = str(plan.get("summary") or "").strip()
        if summary:
            lines.append(f"\n> {summary}\n")

        nodes = [n for n in plan.get("nodes", []) if isinstance(n, dict)]

        # Compact flow line — the whole shape at a glance.
        flow = " → ".join(str(n.get("node", "?")) for n in nodes)
        if flow:
            lines.append(f"**Flow:** {flow}\n")

        lines.append("## 🧩 Build these steps, in order")
        for n in nodes:
            step = n.get("step", "?")
            node = str(n.get("node", "?"))
            cfg = str(n.get("config") or "").strip()
            purpose = str(n.get("purpose") or "").strip()
            lines.append(f"\n**{step}. {self._node_emoji(node)} {node}**")
            if purpose:
                lines.append(f"> {purpose}")
            if cfg:
                lines.append(f"`⚙ {cfg}`")

        # Tools used — what each distinct tool in this plan is for, and whether
        # it needs a connected account. Answers "which tool for which work/access".
        seen: List[str] = []
        for n in nodes:
            nm = str(n.get("node", "")).strip()
            if nm and nm not in seen:
                seen.append(nm)
        if seen:
            lines.append("\n## 🧰 Tools used")
            for nm in seen:
                purpose = _PURPOSE.get(nm)
                key = " · 🔑 needs a connected account" if nm in _NEEDS_CREDENTIAL else ""
                if purpose:
                    lines.append(f"- **{nm}** — {purpose}{key}")
                else:
                    lines.append(f"- **{nm}**{key}")

        notes = str(plan.get("notes") or "").strip()
        if notes:
            lines.append(f"\n## 📌 Notes\n{notes}")

        if check and check.get("checks"):
            status = check.get("status", "pass")
            header = {
                "pass": "## ✅ Break check — all checks passed",
                "warnings": "## ⚠️ Break check — ready, with warnings",
                "breaks": "## ❌ Break check — this workflow will not run",
            }.get(status, "## Break check")
            lines.append("\n" + header)
            icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}
            for c in check["checks"]:
                lines.append(f"- {icon.get(c['level'], '-')} {c['message']}")
        elif warnings:
            lines.append("\n## ⚠️ Validation")
            for w in warnings:
                lines.append(f"- {w}")
        return "\n".join(lines)

    # ---- run ---------------------------------------------------------------

    async def run(
        self,
        user_query: str,
        provided_data: Optional[Any] = None,
        session_id: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Dict, None]:
        try:
            params = self._extract_params(user_query, tool_args)
            goal = params["goal"]

            if not goal:
                out = {"success": False, "error": "No workflow goal/description provided", "result": None}
                yield event(AgentEvent.RESULT, out)
                yield event(AgentEvent.FINAL, {"success": False, "error": out["error"], "response": None})
                return

            # Tools reference: if the user is asking what tools exist (not to design
            # a workflow), answer straight from the catalog — no LLM call needed.
            if _wants_tools_reference(goal):
                reference = _tools_reference()
                out = {"success": True, "result": reference, "response": reference, "tools": True}
                yield event(AgentEvent.RESULT, out)
                yield event(AgentEvent.FINAL, {"success": True, "response": reference, "result": out})
                return

            if not self.llm_provider or not getattr(self.llm_provider, "get_response", None):
                out = {"success": False, "error": "LLM provider not configured", "result": None}
                yield event(AgentEvent.RESULT, out)
                yield event(AgentEvent.FINAL, {"success": False, "error": out["error"], "response": None})
                return

            yield event(AgentEvent.THOUGHT, "Designing workflow from the spec...")

            response = await self.llm_provider.get_response(
                transcript=self._user_prompt(goal, params["inputs"], params["outputs"]),
                system_prompt=self._system_prompt(),
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
            )

            if not response or not str(response).strip():
                out = {"success": False, "error": "No response from LLM", "result": None}
                yield event(AgentEvent.RESULT, out)
                yield event(AgentEvent.FINAL, {"success": False, "error": out["error"], "response": None})
                return

            plan = self._parse_json_loose(response)
            if plan is None:
                # LLM returned prose instead of JSON — pass it through rather than fail.
                out = {
                    "success": True,
                    "result": response.strip(),
                    "response": response.strip(),
                    "structured": None,
                    "warnings": ["Could not parse a structured plan; returned the raw design text."],
                }
                yield event(AgentEvent.RESULT, out)
                yield event(AgentEvent.FINAL, {"success": True, "response": out["response"], "result": out})
                return

            warnings, _ = self._validate(plan, params["extra_nodes"])
            check = self._break_check(plan, params["extra_nodes"])
            markdown = self._render_markdown(plan, warnings, check)

            out = {
                "success": True,
                "result": markdown,
                "response": markdown,
                "structured": plan,
                "warnings": warnings,
                "check": check,
                "status": check["status"],
            }
            yield event(AgentEvent.RESULT, out)
            yield event(AgentEvent.FINAL, {"success": True, "response": markdown, "result": out})

        except Exception as e:
            logger.exception("Workflow Architect tool failed")
            yield event(AgentEvent.ERROR, {"error": str(e)})
            yield event(AgentEvent.FINAL, {"success": False, "error": str(e), "response": None})

    async def initialize(self) -> None:
        logger.debug("Workflow Architect tool initialized")

    async def cleanup(self) -> None:
        logger.debug("Workflow Architect tool cleaned up")


__all__ = ["WorkflowArchitectAgentTool"]
