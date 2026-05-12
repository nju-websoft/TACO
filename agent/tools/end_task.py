from typing import Dict, Any, Optional
from langchain_core.tools import tool
from agent.dispatch import global_dispatcher


@tool("end_task", return_direct=False)
def end_task(status: str, output_path: str, result: Optional[str | Dict[str, Any]] = None) -> str:
    """End the current task and return control to the Planner.
    
    Use this tool when you have completed the assigned task and want to return
    control to the Planner for the next task assignment.
    
    Args:
        status: The status of the task, either "success" or "failure".
        output_path: The absolute path to the refined dataset file.
        result: A dictionary containing any other relevant information about the task completion.
    """
    
    global_dispatcher.emit_tool_call(name="end_task", args={"status": status, "output_path": output_path}, agent="executor")
    return f"Task completed with status: {status}. Output path: {output_path}. " \
           f"Other information: {result}"
