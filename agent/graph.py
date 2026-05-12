from agent.components.executor import UnifiedState, executor, should_continue, emit_tool, emit_continue, monitor_node, cleanup_node
from agent.components.planner import planner_node
from agent.tools import TOOLS
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json

def planner_route(state: UnifiedState):
    """Route after planner: END if all tasks completed, else executor."""
    if state.get("all_tasks_completed"):
        return END
    return "executor"


def custom_tool_node(state: UnifiedState):
    """Custom tool node that auto-injects dataset_dir and auto-updates dataset_info."""
    last_msg = state.get("messages", [])[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])

    # Get dataset_dir from state
    dataset_dir = state.get("dataset_info", {}).get("dir", "")

    # Inject dataset_dir for subagent tools
    if dataset_dir:
        for tc in tool_calls:
            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if tool_name and "_agent" in tool_name:
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                if isinstance(args, dict):
                    args["dataset_dir"] = dataset_dir

    # Execute tools
    tool_node = ToolNode(TOOLS)
    result = tool_node.invoke(state)

    # Auto-update dataset_info from subagent results
    messages = result.get("messages", [])
    dataset_info = dict(state.get("dataset_info") or {})
    updated = False

    for msg in messages:
        if hasattr(msg, "name") and "_agent" in msg.name:
            try:
                content = getattr(msg, "content", "") or ""
                res = json.loads(content)
                if isinstance(res, dict) and res.get("status") == "success" and "output_dir" in res:
                    dataset_info["dir"] = res["output_dir"]
                    updated = True
            except:
                pass

    if updated:
        result["dataset_info"] = dataset_info

    return result

graph = StateGraph(UnifiedState)

# Nodes
graph.add_node("planner", planner_node)
graph.add_node("executor", executor)
graph.add_node("tools", custom_tool_node)
graph.add_node("emit_tool", emit_tool)
graph.add_node("monitor", monitor_node)
graph.add_node("cleanup", cleanup_node)

# Entry Point
graph.set_entry_point("planner")

# Edges
graph.add_conditional_edges(
    "planner",
    planner_route,
    {
        "executor": "executor",
        END: END
    }
)
graph.add_edge("cleanup", "executor")

graph.add_conditional_edges(
    "executor",
    should_continue,
    {
        "tools": "tools",
        "planner": "planner",
        "cleanup": "cleanup",
        "monitor": "monitor",
        "executor": "executor",
        END: END
    }
)

graph.add_edge("tools", "emit_tool")
graph.add_edge("monitor", "executor")

graph.add_conditional_edges(
    "emit_tool",
    emit_continue,
    {
        "again": "executor",
        "stop": END
    }
)

app = graph.compile()