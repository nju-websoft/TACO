"""Unified LangGraph application with tool-calling loop and streaming helpers.

- Defines a unified state, model/tool loop, retry limits, and system prompt.
- Provides stream APIs for CLI/Web rendering, plus history compaction & summarization.
"""

from typing_extensions import TypedDict, Annotated
from typing import List, Optional, Dict, Any
import json
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from agent.model import agent_llm
from agent.tools import TOOLS
from agent.utils import _todo_read
from agent.context import compress_context
from agent.dispatch import global_dispatcher

from agent.tools.subagents_prompt.rough_filter_prompt import EXECUTOR_SYS_PROMPT as rough_filter_executor_prompt
from agent.tools.subagents_prompt.fine_filter_prompt import EXECUTOR_SYS_PROMPT as fine_filter_executor_prompt
from agent.tools.subagents_prompt.base_model_filter_prompt import EXECUTOR_SYS_PROMPT as base_model_filter_executor_prompt
from agent.tools.subagents_prompt.discipline_discovery_prompt import EXECUTOR_SYS_PROMPT as discipline_discovery_executor_prompt


class UnifiedState(TypedDict):
    messages: Annotated[List[object], add_messages]
    tool_request: Optional[Dict[str, Any]]
    tool_result: Optional[str]
    retries: Optional[Dict[str, int]]
    tool_calls_count: Optional[int]
    todos: Optional[List[Dict[str, Any]]]
    dataset_info: Optional[Dict[str, Any]]
    finetune_goal: Optional[str]
    active_task_id: Optional[str]
    bash_count: Optional[int]
    task_start_index: Optional[int]
    all_tasks_completed: Optional[bool]


EXECUTOR_SYS_PROMPT = """You are the Executor of ADCF. You execute one task at a time by calling the assigned Sub-Agent, then report results back to the Planner.

**Current Task**:
- Task: {active_task_id} — {active_task_description}
- Sub-Agent: {sub_agent_type}
- Dataset Directory: {dataset_dir}
- Finetune Goal: {finetune_goal}
- Dataset Profile: {raw_dataset_profile}

**Execution Steps** (follow in order):
1. **Validate parameters**: If unsure about dataset structure or sub-agent params, use bash_exec to inspect (e.g. `head -5 file.json`, `wc -l file.json`, `jq 'keys' file.json`). One command per bash_exec call. Do NOT use `cat` or `grep` on full files. dataset_dir is auto-injected — do not pass it manually.
2. **Call the Sub-Agent**: Once params are confirmed, make a single tool call to {sub_agent_type} with the required arguments.
3. **Report results**: After the Sub-Agent returns, you may use up to {post_subagent_bash_limits} bash_exec calls to verify output (e.g. `wc -l`, `head`). Then return a result dict as plain text:
   {{"status": "success"|"failure", "output_path": "/abs/path", "original_count": N, "filtered_count": M}}

**Constraints**:
- Call the Sub-Agent exactly ONCE. After it returns, do NOT call any Sub-Agent again.
- bash_exec is for inspection only — do not use it to run processing scripts.

**Sub-Agent Parameter Reference ({sub_agent_type})**:
{sub_agent_prompt}
"""

BASH_TOOL_LIMITS=8
POST_SUBAGENT_BASH_LIMITS=4

def executor(state: UnifiedState):
    """Bind tools and invoke the model; capture first tool-call metadata."""

    last_msg = state.get("messages")[-1] if state.get("messages") else None
    
    llm_with_tools = agent_llm.bind_tools(TOOLS)
    
    base_msgs = list(state.get("messages") or [])
    # Sync todos from global storage before execution
    current_todos = _todo_read()

    # Generate explicit system prompt for Executor
    executing_task = next((t for t in current_todos if t.get("status") == "executing"), None)
    active_task_id = executing_task.get("id") if executing_task else None
    active_task_description = executing_task.get("description") if executing_task else None
    sub_agent_type = executing_task.get("sub_agent") if executing_task else None
    retry_hint = executing_task.get("retry_hint", "") if executing_task else ""
    
    global_dispatcher.emit_tool_call(
        name="executor_start",
        args={"task_id": active_task_id, "sub_agent": sub_agent_type},
        agent="executor"
    )

    # State management for bash_exec limits
    previous_task_id = state.get("active_task_id")
    bash_count = state.get("bash_count", 0)
    task_start_index = state.get("task_start_index", 0)
    
    if active_task_id != previous_task_id:
        bash_count = 0
        # task_start_index indexes into state["messages"], must use original length
        task_start_index = len(state.get("messages") or [])
        # Clear old task context: keep only the original user message
        # and the latest planner message (task transition boundary).
        # Search backward for the planner's output: an AIMessage without tool_calls,
        # or a SystemMessage. Avoid picking up executor's own AIMessage (has tool_calls)
        # or ToolMessage from sub-agents.
        last_planner_msg = None
        for _m in reversed(base_msgs[1:]):
            if isinstance(_m, SystemMessage):
                last_planner_msg = _m
                break
            if isinstance(_m, AIMessage) and not getattr(_m, "tool_calls", None):
                last_planner_msg = _m
                break
        if last_planner_msg is None and len(base_msgs) > 1:
            last_planner_msg = base_msgs[-1]
        # Only keep the planner's latest output — the original user instruction
        # is already in exec_sys_prompt as {finetune_goal}
        base_msgs = [last_planner_msg] if last_planner_msg else []
    
    dataset_info = state.get("dataset_info") or {}
    raw_dataset_profile = dataset_info.get("profile") or ""
    dataset_dir = dataset_info.get("dir") or ""
    print("[Executor] EXECUTOR DATASET DIR: ", dataset_dir)

    # Select only the relevant sub-agent prompt to save tokens
    _sub_agent_prompts = {
        "discipline_discovery_agent": discipline_discovery_executor_prompt,
        "rough_filter_agent": rough_filter_executor_prompt,
        "fine_filter_agent": fine_filter_executor_prompt,
        "base_model_filter_agent": base_model_filter_executor_prompt,
    }
    sub_agent_prompt = _sub_agent_prompts.get(sub_agent_type, "No parameter reference available.")

    exec_sys_prompt = SystemMessage(content=EXECUTOR_SYS_PROMPT.format(
                                                                active_task_id=active_task_id,
                                                                active_task_description=active_task_description,
                                                                sub_agent_type=sub_agent_type,
                                                                raw_dataset_profile=raw_dataset_profile,
                                                                dataset_dir=dataset_dir,
                                                                finetune_goal=state.get("finetune_goal") or "No finetune goal specified.",
                                                                sub_agent_prompt=sub_agent_prompt,
                                                                post_subagent_bash_limits=POST_SUBAGENT_BASH_LIMITS))
    # Only compress if history is large enough to warrant it
    if base_msgs and sum(len(getattr(m, "content", "") or "") for m in base_msgs) > 15000:
        history_msgs = compress_context(base_msgs)
    else:
        history_msgs = list(base_msgs)
    msgs = history_msgs + [exec_sys_prompt]

    if isinstance(last_msg, ToolMessage) and last_msg.name == "bash_exec":
        msgs.append(SystemMessage(content="You have used bash_exec for inspection. If you now have enough information about the dataset, call the assigned Sub-Agent with the validated parameters."))
    if retry_hint:
        msgs.append(SystemMessage(content=retry_hint))
        
    # Inject workflow configuration
    workflow_mode = state.get("workflow_mode")
    fine_filter_dims = state.get("fine_filter_dimensions")
    if workflow_mode and fine_filter_dims and sub_agent_type and "fine_filter" in sub_agent_type:
        config_hint = f"WORKFLOW CONFIG: Mode={workflow_mode}. Use these dimensions for fine_filter_config: {fine_filter_dims}"
        msgs.append(SystemMessage(content=config_hint))

    resp = llm_with_tools.invoke(msgs)

    tr = None
    tcs = getattr(resp, "tool_calls", None)
    if tcs:
        tc = tcs[0]
        if isinstance(tc, dict):
            tr = {"name": tc.get("name"), "args": tc.get("args")}
        else:
            tr = {"name": getattr(tc, "name", None), "args": getattr(tc, "args", None)}
            
        # Increment bash_count if bash_exec is called
        for t in tcs:
            t_name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
            if t_name == "bash_exec":
                bash_count += 1
                
    retries = state.get("retries") or {}
    calls = int(state.get("tool_calls_count", 0))
    if tcs:
        calls += 1
        
    state_diff = {
        "messages": [resp], 
        "tool_request": tr, 
        "retries": retries, 
        "tool_calls_count": calls, 
        "active_task_id": active_task_id, 
        "bash_count": bash_count, 
        "task_start_index": task_start_index
    }
    
    return state_diff


def monitor_node(state: UnifiedState):
    """Monitor node to intercept incorrect tool calls."""

    
    last_msg = state.get("messages", [])[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])
    
    todos = state.get("todos") or []
    executing_task = next((t for t in todos if t.get("status") == "executing"), None)
    target_agent = executing_task.get("sub_agent") if executing_task else "Unknown"
    
    # Use task_start_index to check history for CURRENT task only
    start_idx = state.get("task_start_index", 0)
    history_messages = state.get("messages", [])[start_idx:-1]
    
    already_called = False
    for msg in history_messages:
        if isinstance(msg, ToolMessage) and msg.name == target_agent:
            already_called = True
            break

    msgs = []
    for tc in tool_calls:
        err_msg = ""
        t_name = tc["name"]
        
        # Mismatch error
        if "_agent" in t_name and t_name != target_agent:
            err_msg = f"Error: You are assigned to use '{target_agent}', but you called '{t_name}'. Please return control to the Planner."
        # Duplicate call error (includes post-subagent re-call attempts)
        elif t_name == target_agent and already_called:
            err_msg = f"Error: The Sub-Agent '{target_agent}' has already been executed successfully in this task. You cannot call it again. Please compile the Sub-Agent results into a feedback dict and return control to the Planner."
        elif "_agent" in t_name and already_called:
            err_msg = f"Error: A Sub-Agent has already been executed in this task. You cannot call any agent again. Please compile the results and return control to the Planner."
        # Bash exec limit error
        elif t_name == "bash_exec":
            err_msg = f"Error: You have reached the maximum limit of {BASH_TOOL_LIMITS} bash_exec calls for this task. Please proceed with the Sub-Agent or return control to the Planner."
        else:
             # Fallback or other errors
             err_msg = f"Error: Invalid tool usage detected for '{t_name}'."

        global_dispatcher.emit_tool_call(
            name="monitor_reject",
            args={"tool": t_name, "reason": err_msg[:80]},
            agent="monitor"
        )
        msgs.append(ToolMessage(
            tool_call_id=tc["id"],
            name=tc["name"],
            content=err_msg
        ))
    return {"messages": msgs}


from langchain_core.messages import RemoveMessage

def cleanup_node(state: UnifiedState):
    """Remove the last AI message (rejected tool call) and inject a reminder to return results."""
    global_dispatcher.emit_tool_call(name="cleanup", args={"task_id": state.get("active_task_id")}, agent="executor")

    messages = state.get("messages", [])
    if messages:
        return {"messages": [
            RemoveMessage(id=messages[-1].id),
            HumanMessage(content="The Sub-Agent has already completed and you have exhausted your post-execution bash allowance. "
                "You MUST NOW return a result dict with 'status' key (e.g. {\"status\": \"success\", \"output_dir\": \"...\", \"original_count\": N, \"filtered_count\": M}) and other necessary descriptions"
                "to the Planner. Do NOT call any more tools.")
        ]}
    return {}


def _count_post_subagent_bash(messages, target_agent):
    """Count bash_exec ToolMessages that appear AFTER the target subagent ToolMessage."""
    found_subagent = False
    count = 0
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == target_agent:
            found_subagent = True
            continue
        if found_subagent and isinstance(m, ToolMessage) and m.name == "bash_exec":
            count += 1
    return count


def should_continue(state: UnifiedState):
    """Decide if the agent should use tool or back to planner for next action."""
    last = state["messages"][-1] if state.get("messages") else None
    if last is None:
        return END
    todos = state.get("todos") or []
    
    # Check current status
    executing = any(str(it.get("status")) == "executing" for it in todos)
    has_pending = any(str(it.get("status")) == "pending" for it in todos)

    # If executing, check if executor wants to update status
    if executing:
        # Check if subagent was used in the CURRENT task
        start_idx = state.get("task_start_index", 0)
        current_messages = state.get("messages", [])[start_idx:]
        
        executing_task = next((t for t in todos if t.get("status") == "executing"), None)
        target_agent = executing_task.get("sub_agent") if executing_task else None
        
        sub_agent_called = False
        if target_agent:
            sub_agent_called = any(
                isinstance(m, ToolMessage) and m.name == target_agent 
                for m in current_messages
            )

        if sub_agent_called:
            # Subagent already executed; check what the executor wants to do next
            if not state.get("tool_request"):
                # No tool call — executor produced a text response, return to planner
                return "planner"

            # Executor wants to call a tool after subagent
            tc = getattr(last, "tool_calls", None) or []
            for t in tc:
                t_name = t.get("name") if isinstance(t, dict) else getattr(t, "name", "")
                # Block any subagent re-call
                if "_agent" in t_name:
                    return "monitor"

            # Allow bash_exec up to POST_SUBAGENT_BASH_LIMITS
            post_bash = _count_post_subagent_bash(current_messages, target_agent)
            wants_bash = any(
                (t.get("name") if isinstance(t, dict) else getattr(t, "name", "")) == "bash_exec"
                for t in tc
            )
            if wants_bash and post_bash < POST_SUBAGENT_BASH_LIMITS:
                return "tools"

            # Limit reached or non-bash tool — force cleanup
            return "cleanup"
        
        content = getattr(last, "content", "") or ""
        if isinstance(last, AIMessage) and "status" in content.lower() and "success" in content.lower():
            return "planner"

    # Check for tool_calls in message OR tool_request in state
    tc = getattr(last, "tool_calls", None)
    tr = state.get("tool_request")
    
    if tc or tr:
        # Check for sub-agent mismatch or duplicate calls
        executing_task = next((t for t in todos if t.get("status") == "executing"), None)
        target_agent = executing_task.get("sub_agent") if executing_task else None
 
        if tc:
            # Check history for previous successful calls of the target agent
            already_called = False
            bash_exec_count = state.get("bash_count", 0)

            if target_agent:
                # Only check history for the CURRENT task using task_start_index
                start_idx = state.get("task_start_index", 0)
                current_messages = state.get("messages", [])[start_idx:-1]
                
                for msg in current_messages:
                    if isinstance(msg, ToolMessage) and msg.name == target_agent:
                        already_called = True
                        break

            for t in tc:
                t_name = t.get("name")
                # 1. Mismatch check
                if "_agent" in t_name and target_agent and t_name != target_agent:
                    return "monitor"
                # 2. Duplicate call check
                if t_name == target_agent and already_called:
                    return "monitor"
                # 3. Bash exec limit check
                if t_name == "bash_exec" and bash_exec_count > BASH_TOOL_LIMITS:
                    return "monitor"
                    
        return "tools"
        
    # If no task is in progress but there are pending tasks, route to planner for assignment
    if not executing and has_pending:
        return "planner"
        
    return END


def emit_tool(state: UnifiedState):
    """Emit tool_result and track empty-result retries per tool name."""
    last = state["messages"][-1] if state.get("messages") else None
    if isinstance(last, ToolMessage):
        content = getattr(last, "content", "") or ""
        func_call_msg = state.get("messages")[-2] if state.get("messages") else None
        tool_name = getattr(func_call_msg, "name", None) if func_call_msg else None
        return {"tool_result": f"Called tool: {tool_name} with args: {func_call_msg.content}, and got result: {content}", "tool_request": None}
    return {"tool_request": None}


def emit_continue(state: UnifiedState):
    """Stop if empty result reached max retries; otherwise loop again."""
    return "again"
