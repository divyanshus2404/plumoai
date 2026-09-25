"""
Self-contained tests for the Workflow Architect agent.

No third-party dependencies: run with either

    python3 -m unittest ai-agents/workflow_architect/test_workflow_architect.py

or directly

    python3 ai-agents/workflow_architect/test_workflow_architect.py

The platform base class (backend.services.ai_agents.base_tool_agent.BaseToolAgent)
is stubbed so the plugin can be imported and exercised outside a running
PlumoAI deployment. Everything else is the real plugin code.
"""

import asyncio
import importlib.util
import os
import sys
import types
import unittest


def _load_tool_module():
    """Stub the closed base class, then import the real tool module by path."""
    for name in (
        "backend",
        "backend.services",
        "backend.services.ai_agents",
        "backend.services.ai_agents.base_tool_agent",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))

    class _BaseToolAgent:  # minimal stand-in for the platform base class
        pass

    sys.modules["backend.services.ai_agents.base_tool_agent"].BaseToolAgent = _BaseToolAgent

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "workflow_architect_agent_tool.py")
    spec = importlib.util.spec_from_file_location("_wa_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wa = _load_tool_module()


class FakeLLM:
    """Mimics the platform's llm_provider. Records what it was asked and returns a canned plan."""

    def __init__(self, response):
        self._response = response
        self.last_system_prompt = None
        self.last_transcript = None

    async def get_response(self, transcript, system_prompt=None, max_tokens=None, temperature=None):
        self.last_system_prompt = system_prompt
        self.last_transcript = transcript
        return self._response


def _run(coro):
    return asyncio.run(coro)


async def _collect(agent, **kwargs):
    """Drive the run() async generator and return (event_types, final_content)."""
    types_seen, final = [], None
    async for ev in agent.run(**kwargs):
        types_seen.append(ev["type"])
        if ev["type"] == "final":
            final = ev["content"]
    return types_seen, final


class NormalizationTests(unittest.TestCase):
    def test_norm_strips_non_alnum(self):
        self.assertEqual(wa._norm("IF Else"), "ifelse")
        self.assertEqual(wa._norm("  Schedule-Trigger  "), "scheduletrigger")

    def test_catalog_text_lists_real_nodes(self):
        text = wa._catalog_text()
        for node in ("Schedule Trigger", "Split Out", "Filter", "Switch", "End Workflow"):
            self.assertIn(node, text)


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.agent = wa.WorkflowArchitectAgentTool(llm_provider=None)

    def test_alias_resolution(self):
        plan = {"nodes": [{"step": 1, "node": "schedule", "category": "Trigger"},
                          {"step": 2, "node": "if", "category": "Flow / Logic"}]}
        warnings, nodes = self.agent._validate(plan, extra_nodes=[])
        self.assertEqual(nodes[0]["node"], "Schedule Trigger")
        self.assertEqual(nodes[1]["node"], "IF Else")
        self.assertEqual(warnings, [])

    def test_unknown_node_flagged(self):
        plan = {"nodes": [{"step": 1, "node": "Manual Trigger", "category": "Trigger"},
                          {"step": 2, "node": "Frobnicate"}]}
        warnings, _ = self.agent._validate(plan, extra_nodes=[])
        self.assertTrue(any("Frobnicate" in w for w in warnings))

    def test_extra_nodes_extend_catalog(self):
        plan = {"nodes": [{"step": 1, "node": "Manual Trigger", "category": "Trigger"},
                          {"step": 2, "node": "HubSpot"}]}
        warnings, _ = self.agent._validate(plan, extra_nodes=["HubSpot"])
        self.assertFalse(any("HubSpot" in w for w in warnings))

    def test_missing_trigger_warned(self):
        plan = {"nodes": [{"step": 1, "node": "Filter", "category": "Data Transformation"}]}
        warnings, _ = self.agent._validate(plan, extra_nodes=[])
        self.assertTrue(any("trigger" in w.lower() for w in warnings))

    def test_empty_nodes_warned(self):
        warnings, nodes = self.agent._validate({"nodes": []}, extra_nodes=[])
        self.assertEqual(nodes, [])
        self.assertTrue(any("no nodes" in w.lower() for w in warnings))


class BreakCheckTests(unittest.TestCase):
    def setUp(self):
        self.agent = wa.WorkflowArchitectAgentTool(llm_provider=None)

    def test_clean_workflow_passes(self):
        plan = {"nodes": [
            {"step": 1, "node": "Schedule Trigger", "category": "Trigger"},
            {"step": 2, "node": "Filter", "category": "Data Transformation"},
        ]}
        r = self.agent._break_check(plan, extra_nodes=[])
        self.assertEqual(r["status"], "pass")
        self.assertTrue(any(c["level"] == "ok" and "one trigger" in c["message"] for c in r["checks"]))

    def test_no_trigger_is_a_break(self):
        plan = {"nodes": [{"step": 1, "node": "Filter", "category": "Data Transformation"}]}
        r = self.agent._break_check(plan, extra_nodes=[])
        self.assertEqual(r["status"], "breaks")
        self.assertTrue(any(c["level"] == "error" and "never start" in c["message"] for c in r["checks"]))

    def test_unknown_node_is_a_break(self):
        plan = {"nodes": [
            {"step": 1, "node": "Manual Trigger", "category": "Trigger"},
            {"step": 2, "node": "MagicMatcher", "category": "AI Agent"},
        ]}
        r = self.agent._break_check(plan, extra_nodes=[])
        self.assertEqual(r["status"], "breaks")
        self.assertTrue(any("MagicMatcher" in c["message"] and c["level"] == "error" for c in r["checks"]))

    def test_credential_node_is_a_warning_not_break(self):
        plan = {"nodes": [
            {"step": 1, "node": "Schedule Trigger", "category": "Trigger"},
            {"step": 2, "node": "Google Sheets", "category": "AI Agent"},
        ]}
        r = self.agent._break_check(plan, extra_nodes=[])
        self.assertEqual(r["status"], "warnings")
        self.assertTrue(any(c["level"] == "warning" and "Google Sheets" in c["message"] for c in r["checks"]))

    def test_trailing_switch_warns(self):
        plan = {"nodes": [
            {"step": 1, "node": "Webhook Trigger", "category": "Trigger"},
            {"step": 2, "node": "Switch", "category": "Flow / Logic"},
        ]}
        r = self.agent._break_check(plan, extra_nodes=[])
        self.assertTrue(any(c["level"] == "warning" and "branch" in c["message"] for c in r["checks"]))

    def test_check_appears_in_run_output(self):
        import asyncio as _a
        plan_json = ('{"workflow_name":"W","nodes":['
                     '{"step":1,"node":"schedule","category":"Trigger"},'
                     '{"step":2,"node":"Google Sheets","category":"AI Agent"}]}')
        agent = wa.WorkflowArchitectAgentTool(llm_provider=FakeLLM(plan_json))
        _, final = _run(_collect(agent, user_query="build it", tool_args=None))
        self.assertIn("check", final["result"])
        self.assertEqual(final["result"]["status"], "warnings")
        self.assertIn("Break check", final["result"]["response"])


class ToolCatalogTests(unittest.TestCase):
    def test_reference_lists_categories_and_key_marker(self):
        ref = wa._tools_reference()
        for cat in ("Triggers", "Data", "AI & Reasoning", "Communication", "Data & CRM"):
            self.assertIn(cat, ref)
        self.assertIn("🔑", ref)                      # credential marker present
        self.assertIn("Google Sheets", ref)
        self.assertIn("Web Search", ref)

    def test_only_installed_tools(self):
        # Slack and GitHub are NOT installed in this setup — must not appear.
        self.assertNotIn("Slack", wa._ALL_TOOL_NAMES)
        self.assertNotIn("GitHub", wa._ALL_TOOL_NAMES)
        # real installed ones are present
        for t in ("Gmail", "Apify", "SendGrid", "WhatsApp Business", "Upwork"):
            self.assertIn(t, wa._ALL_TOOL_NAMES)

    def test_credentials_derived_correctly(self):
        self.assertIn("Gmail", wa._NEEDS_CREDENTIAL)
        self.assertIn("Google Sheets", wa._NEEDS_CREDENTIAL)
        self.assertNotIn("Filter", wa._NEEDS_CREDENTIAL)      # built-in node, no account
        self.assertNotIn("AI Agent", wa._NEEDS_CREDENTIAL)

    def test_wants_tools_reference_intent(self):
        for yes in ("what tools are available?", "which tool for sending email", "list tools"):
            self.assertTrue(wa._wants_tools_reference(yes), yes)
        long_spec = "every morning search jobs, dedupe, filter, and log them to a sheet " * 3
        self.assertFalse(wa._wants_tools_reference(long_spec))

    def test_run_returns_tools_reference_without_llm(self):
        # No llm_provider needed — the reference is static.
        agent = wa.WorkflowArchitectAgentTool(llm_provider=None)
        _, final = _run(_collect(agent, user_query="what tools are available?", tool_args=None))
        self.assertTrue(final["success"])
        self.assertTrue(final["result"].get("tools"))
        self.assertIn("Available tools", final["result"]["response"])

    def test_design_output_has_tools_used_section(self):
        plan_json = ('{"workflow_name":"W","nodes":['
                     '{"step":1,"node":"Schedule Trigger","category":"Trigger"},'
                     '{"step":2,"node":"Gmail","category":"AI Agent"}]}')
        agent = wa.WorkflowArchitectAgentTool(llm_provider=FakeLLM(plan_json))
        _, final = _run(_collect(agent, user_query="build it", tool_args=None))
        resp = final["result"]["response"]
        self.assertIn("Tools used", resp)
        self.assertIn("🔑", resp)   # Gmail needs an account


class JsonParsingTests(unittest.TestCase):
    def setUp(self):
        self.agent = wa.WorkflowArchitectAgentTool(llm_provider=None)

    def test_parses_fenced_json_with_trailing_prose(self):
        plan = self.agent._parse_json_loose('```json\n{"workflow_name":"X","nodes":[]}\n``` all done')
        self.assertIsNotNone(plan)
        self.assertEqual(plan["workflow_name"], "X")

    def test_returns_none_on_garbage(self):
        self.assertIsNone(self.agent._parse_json_loose("no json here at all"))


class RunTests(unittest.TestCase):
    def _agent_with_plan(self, plan_json):
        return wa.WorkflowArchitectAgentTool(llm_provider=FakeLLM(plan_json))

    def test_run_end_to_end_success_and_grounding(self):
        plan_json = (
            '{"workflow_name":"Daily Job Hunt","summary":"s",'
            '"nodes":[{"step":1,"node":"schedule","category":"Trigger","config":"09:00","purpose":"start"},'
            '{"step":2,"node":"Filter","category":"Data Transformation","config":"score>=70","purpose":"keep"}],'
            '"connections":"1 -> 2"}'
        )
        agent = self._agent_with_plan(plan_json)
        spec = '{"goal":"find jobs for my roles","inputs":"roles, location","outputs":"sheet rows"}'
        types_seen, final = _run(_collect(agent, user_query=spec, tool_args=None))

        self.assertIn("final", types_seen)
        self.assertTrue(final["success"])
        # the real node catalog must have been injected into the system prompt (grounding)
        self.assertIn("Schedule Trigger", agent.llm_provider.last_system_prompt)
        # the user's spec must have reached the model
        self.assertIn("roles", agent.llm_provider.last_transcript.lower())
        # alias normalization happened in the structured output
        self.assertEqual(final["result"]["structured"]["nodes"][0]["node"], "Schedule Trigger")
        # rendered markdown is present and readable
        self.assertIn("# Daily Job Hunt", final["result"]["result"])

    def test_run_flags_bogus_node_from_model(self):
        plan_json = (
            '{"workflow_name":"W","nodes":['
            '{"step":1,"node":"Manual Trigger","category":"Trigger"},'
            '{"step":2,"node":"MagicMatcher","category":"AI Agent"}]}'
        )
        agent = self._agent_with_plan(plan_json)
        _, final = _run(_collect(agent, user_query="build something", tool_args=None))
        self.assertTrue(final["success"])
        self.assertTrue(any("MagicMatcher" in w for w in final["result"]["warnings"]))

    def test_run_without_goal_fails_cleanly(self):
        agent = self._agent_with_plan("{}")
        _, final = _run(_collect(agent, user_query="", tool_args=None))
        self.assertFalse(final["success"])
        self.assertIn("goal", final["error"].lower())

    def test_run_without_llm_provider_fails_cleanly(self):
        agent = wa.WorkflowArchitectAgentTool(llm_provider=None)
        _, final = _run(_collect(agent, user_query="build a workflow", tool_args=None))
        self.assertFalse(final["success"])
        self.assertIn("llm provider", final["error"].lower())

    def test_run_passes_through_non_json_model_output(self):
        agent = self._agent_with_plan("Here is a plan in prose, not JSON.")
        _, final = _run(_collect(agent, user_query="build a workflow", tool_args=None))
        self.assertTrue(final["success"])  # graceful: return the prose rather than error
        self.assertIsNone(final["result"]["structured"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
