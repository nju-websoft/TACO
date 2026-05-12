from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.model import agent_llm
from agent.utils import _todo_read, _todo_write, get_clean_content
from agent.tools.subagents_prompt.rough_filter_prompt import PLANNER_SYS_PROMPT as rough_filter_planner_prompt
from agent.tools.subagents_prompt.fine_filter_prompt import PLANNER_SYS_PROMPT as fine_filter_planner_prompt
from agent.tools.subagents_prompt.base_model_filter_prompt import PLANNER_SYS_PROMPT as base_model_filter_planner_prompt
from agent.tools.subagents_prompt.discipline_discovery_prompt import PLANNER_SYS_PROMPT as discipline_discovery_planner_prompt
from agent.dispatch import global_dispatcher
from agent.config import load_config
import json
import os
from agent.workflow_modes import detect_workflow_mode, get_fine_filter_dimensions, get_workflow_order, WorkflowMode


PLANNER_PROMPT = """You are the Planner of ADCF. Your job is to create and maintain a task roadmap for dataset curation.

**Workflow Mode**: {workflow_mode}
**Recommended Task Order**: {workflow_order}
**Fine-Filter Dimensions**: {fine_filter_dimensions}

**Context**:
- Dataset Profile: {raw_dataset_profile}
- Goal: {goal}
- Current Roadmap: {current_todos}
- Executor Feedback: {execution_feedback}

**Rules**:
1. Follow the recommended task order above. Each task uses exactly one sub_agent.
2. Exactly ONE task may be "executing" at a time. Others are "pending" or "completed".
3. When the Executor reports success for a task, mark it "completed" and set the next pending task to "executing".
4. When the Executor reports failure, decide whether to retry (keep status "executing" and add a retry_hint) or skip (mark "completed" and move on).
5. When ALL tasks are "completed", the pipeline is finished.

**Output**: Return a JSON object with this exact structure:
{{
  "strategic_overview": "One sentence summarizing current pipeline status and next action",
  "workflow_mode": "{workflow_mode}",
  "global_roadmap": [
    {{
      "id": "T1",
      "stage": "rough_filter",
      "name": "Descriptive task name",
      "sub_agent": "rough_filter_agent",
      "status": "pending|executing|completed",
      "description": "What this task does and key parameters"
    }}
  ]
}}

Return ONLY the JSON object. Do not include any other text.
"""

dataset_dir = os.path.join(os.getcwd(), "dataset/gsm8k")

goal = ""
raw_dataset_profile = ""

def set_dataset_dir(path: str):
    """Allow external callers (e.g. run_pipeline) to configure dataset_dir."""
    global dataset_dir
    dataset_dir = os.path.abspath(path)


def planner_node(state):
    """Planner node: Analyzes state, updates todos, and guides the executor."""
    global_dispatcher.emit_tool_call(name="planner_start", args={}, agent="planner")
    
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    
    # 0. Validate rough_agent results with LLM
    if last_msg and isinstance(last_msg, AIMessage):
        feedback_content = getattr(last_msg, "content", "") or ""
        current_todos = _todo_read()
        executing_task = next((t for t in current_todos if t.get("status") == "executing"), None)

        if executing_task and executing_task.get("sub_agent") == "rough_filter_agent":
            min_retention = load_config().get("rough_filter_min_retention", 0.8)
            min_retention_pct = int(min_retention * 100)
            validation_prompt = f"""You are validating the result of a rough_filter_agent execution.

Executor Feedback:
{feedback_content}

Task: Determine if the rough_filter was too strict and needs to be retried with relaxed conditions.
Criteria: If the filter retained less than {min_retention_pct}% of the original data, it should be retried.

Respond with ONLY "True" (needs retry) or "False" (acceptable result)."""

            try:
                validation_response = agent_llm.invoke([SystemMessage(content=validation_prompt)])
                decision = getattr(validation_response, "content", "").strip().lower()

                if "true" in decision:
                    # Extract output_dir from dataset_info and reset to parent
                    dataset_info = dict(state.get('dataset_info') or {})
                    output_dir = dataset_info.get('dir', '')
                    # Safety: only proceed if rough_filter actually produced output
                    # (dataset_info['dir'] was updated to a rough output dir).
                    # If it still points to the original dataset, rough_filter
                    # must have failed — skip deletion entirely.
                    if not output_dir or 'rough' not in os.path.basename(output_dir):
                        print(f'[PLANNER] rough_filter did not produce output (dir={output_dir}), skipping retry deletion.')
                        executing_task["status"] = "executing"
                        executing_task["retry_hint"] = "Previous rough_filter failed. Please inspect the dataset and retry."
                        _todo_write(current_todos)
                        return {
                            "messages": [SystemMessage(content="Rough filter failed (no output dir). Retrying without deletion.")],
                            "todos": current_todos,
                            "dataset_info": dataset_info,
                            "task_start_index": len(state.get("messages") or []),
                            "bash_count": 0,
                        }
                        
                    if output_dir and os.path.isdir(output_dir) and 'rough' in os.path.basename(output_dir):
                        import shutil
                        shutil.rmtree(output_dir)
                    elif output_dir and os.path.isdir(output_dir):
                        print(f'[PLANNER] WARNING: Skipped deletion of {output_dir} (not a rough filter output directory)')

                    # Reset task with retry hint
                    executing_task["status"] = "executing"
                    executing_task["retry_hint"] = f"Previous rough_filter was too strict (< {min_retention_pct}% retention). Please relax filtering conditions."
                    _todo_write(current_todos)

                    parent_dir = os.path.dirname(output_dir) if output_dir else ''
                    if parent_dir:
                        dataset_info['dir'] = parent_dir
                    print(f"[PLANNER] Reset dataset dir to {parent_dir}")
                    
                    # Reset task_start_index so executor sees a clean history window
                    # (otherwise should_continue still sees the old subagent call and blocks re-invocation)
                    new_start = len(state.get("messages") or [])
                    return {
                        "messages": [SystemMessage(content="Rough filter too strict. Output deleted. Retry with relaxed conditions.")],
                        "todos": current_todos,
                        "dataset_info": dataset_info,
                        "task_start_index": new_start,
                        "bash_count": 0,
                    }
            except:
                pass

    # 1. Gather Context
    execution_feedback = "Have not utilized executor yet."
    if isinstance(last_msg, HumanMessage):
        goal = last_msg.content
        state["finetune_goal"] = goal
        # Detect workflow mode from instruction keywords
        workflow_mode = detect_workflow_mode(goal)
        workflow_order = get_workflow_order(workflow_mode)
        fine_filter_dims = get_fine_filter_dimensions(workflow_mode, goal)
        state["workflow_mode"] = workflow_mode
        state["workflow_order"] = workflow_order
        state["fine_filter_dimensions"] = fine_filter_dims
    else:
        execution_feedback = last_msg.content
        goal = state.get("finetune_goal") or ""
        workflow_mode = state.get("workflow_mode", WorkflowMode.SINGLE_DISCIPLINE_QUALITY)
        workflow_order = state.get("workflow_order", ["rough_filter", "base_model_filter", "fine_filter"])
        fine_filter_dims = state.get("fine_filter_dimensions", get_fine_filter_dimensions(workflow_mode))

    # Use compressed context to save tokens, but ensure we have enough info
    
    current_todos = _todo_read()
    todo_str = json.dumps(current_todos, indent=2) if current_todos else "No todos yet."

    # 2. Invoke Planner LLM
    with open(os.path.join(dataset_dir, "profile.txt"), "r") as f:
        raw_dataset_profile = f.read()
        
    prompt = PLANNER_PROMPT.format(
        workflow_mode=workflow_mode,
        workflow_order=str(workflow_order),
        fine_filter_dimensions=str(fine_filter_dims),
        raw_dataset_profile=raw_dataset_profile,
        goal=goal,
        current_todos=todo_str,
        execution_feedback=execution_feedback,
    )
    
    # Planner only needs the prompt (which already contains goal, feedback, todos, profile).
    # No need to pass full message history — executor feedback is in {execution_feedback}.
    planner_messages = [SystemMessage(content=prompt)]
    
    try:
        response = agent_llm.invoke(planner_messages)
        content = getattr(response, "content", "") or "{}"
        plan_data = get_clean_content(content)
        
        # 3. Update Todos
        new_todos = plan_data.get("global_roadmap", [])
        if new_todos:
            # Validate status values
            for t in new_todos:
                if t.get("status") not in ["pending", "executing", "completed"]:
                    t["status"] = "pending"
            
            # Ensure exactly one task is executing (unless all completed)
            executing_tasks = [t for t in new_todos if t.get("status") == "executing"]
            completed_tasks = [t for t in new_todos if t.get("status") == "completed"]
            
            # If no executing task and there are pending tasks, activate the first pending one
            if len(executing_tasks) == 0 and len(completed_tasks) < len(new_todos):
                for t in new_todos:
                    if t.get("status") == "pending":
                        t["status"] = "executing"
                        break
            
            # If multiple executing tasks, keep only the first one as executing
            elif len(executing_tasks) > 1:
                executing_tasks[0]["status"] = "executing"
                for t in executing_tasks[1:]:
                    t["status"] = "pending"
            
            _todo_write(new_todos)

        # Emit planner progress summary
        executing_t = next((t for t in new_todos if t.get("status") == "executing"), None)
        completed_count = sum(1 for t in new_todos if t.get("status") == "completed")
        global_dispatcher.emit_tool_call(
            name="planner_done",
            args={"total_tasks": len(new_todos), "completed": completed_count,
                  "executing": executing_t.get("id") if executing_t else None},
            agent="planner"
        )

        # Check if all tasks are completed
        all_completed = (
            len(new_todos) > 0
            and all(t.get("status") == "completed" for t in new_todos)
        )

        state_diff = {
            "todos": new_todos,
        }

        if all_completed:
            state_diff["all_tasks_completed"] = True
            summary = plan_data.get("strategic_overview", "All filtering tasks completed.")
            state_diff["messages"] = [
                AIMessage(content=f"[Planner] All tasks completed. {summary}")
            ]
            print(f"[PLANNER] All tasks completed — exiting agent system.")
            return state_diff
        # Preserve task context cache across planner transitions
        if state.get("task_context_cache"):
            state_diff["task_context_cache"] = state.get("task_context_cache")
        
        if isinstance(last_msg, HumanMessage):
            state_diff["dataset_info"] = {
                "dir": dataset_dir,
                "profile": raw_dataset_profile,
            }
            state_diff["finetune_goal"] = goal
            state_diff["workflow_mode"] = workflow_mode
            state_diff["workflow_order"] = workflow_order
            state_diff["fine_filter_dimensions"] = fine_filter_dims
        return state_diff

    except Exception as e:
        # Fallback if planner fails
        return {
            "messages": [SystemMessage(content=f"*** PLANNER ERROR ***\nCould not generate plan: {str(e)}\nPlease continue with existing tasks.")],
        }
