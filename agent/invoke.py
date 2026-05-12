from typing import List
from langchain_core.messages import AIMessage
from .graph import app
from .context import compress_context


SYSTEM_PROMPT = """
**Autonomous Data Curation Framework (ADCF)**

You are an automated data curation system that refines raw datasets into high-quality training data for LLMs. The system has two modules that alternate control:

**Planner** — Strategic planning and state management.
- Reads the dataset profile and user goal to create a task roadmap (global_roadmap).
- Receives execution feedback after each task and decides: advance to next task, retry, or finish.
- Only the Planner can change task statuses or reorder the roadmap.

**Executor** — Task execution and tool orchestration.
- Takes the single "executing" task from the roadmap, calls the appropriate Sub-Agent with validated parameters, and returns a structured result report to the Planner.
- Has access to bash_exec for parameter validation and Sub-Agents (rough_filter, fine_filter, base_model_filter) for data processing.

**Workflow**:
1. Planner creates the roadmap and sets the first task to "executing".
2. Executor runs the task via Sub-Agent, returns a result dict to the Planner.
3. Planner marks the task "completed", activates the next one. Repeat until all tasks are done.

**Principles**: Prioritize data quality over quantity. Maintain diversity to avoid domain collapse.
"""

def stream_with_history(messages: List[object]):
    """Stream with history compaction & role/content normalization for Web UI."""
    from .dispatch import global_dispatcher
    import queue
    import threading

    msgs = compress_context(messages)
    
    init = {"messages": msgs}
    
    # Queue to merge events from app stream and global dispatcher
    q = queue.Queue()
    
    def on_dispatch(etype, data):
        q.put((etype, data))
        
    global_dispatcher.register(on_dispatch)
    
    def run_app():
        try:
            for event in app.stream(init, {"recursion_limit": 500}):
                q.put(("app_event", event))
            q.put(("done", None))
        except Exception as e:
            q.put(("error", e))
            
    t = threading.Thread(target=run_app)
    t.start()
    
    try:
        while True:
            try:
                # Poll with timeout to allow checking for thread liveness if needed
                etype, data = q.get(timeout=0.1)
                
                if etype == "done":
                    break
                if etype == "error":
                    # In a real app we might log this or re-raise
                    # For now, break to stop streaming
                    break
                    
                if etype == "tool_call":
                    name = data.get("name")
                    args = data.get("args") or {}
                    agent = data.get("agent") or ""
                    prefix = f"{agent}/" if agent else ""

                    if "current" in args and "total" in args:
                        cur, tot, pct = args["current"], args["total"], args.get("percent", 0)
                        extra = {k: v for k, v in args.items() if k not in ("current", "total", "percent")}
                        extra_s = f" | {extra}" if extra else ""
                        yield f"[progress] {prefix}{name} {cur}/{tot} ({pct}%){extra_s}"
                    elif name.endswith("_done"):
                        detail = " | ".join(f"{k}={v}" for k, v in args.items())
                        yield f"[done] {prefix}{name}: {detail}"
                    else:
                        detail = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
                        yield f"[tool_call] {prefix}{name}({detail})"
                    
                elif etype == "app_event":
                    # Standard app stream event
                    event = data
                    if not isinstance(event, dict):
                        continue
                    for _, values in event.items():
                        if not isinstance(values, dict):
                            continue
                        tr = values.get("tool_request")
                        if tr:
                            name = tr.get("name")
                            args = tr.get("args")
                            yield f"[tool_call] {name}({args})"
                        todos = values.get("todos")
                        if isinstance(todos, list) and todos:
                            lines = [f"[todo] {it.get('id')}. [{it.get('status')}] {it.get('name')}: {it.get('description')}" for it in todos]
                            yield "\n".join(lines)
                        messages_list = values.get("messages")
                        if messages_list:
                            last = messages_list[-1]
                            if isinstance(last, AIMessage):
                                tcs = getattr(last, "tool_calls", None)
                                if tcs:
                                    continue
                                yield getattr(last, "content", str(last))
            except queue.Empty:
                continue
    finally:
        # Cleanup
        global_dispatcher.unregister(on_dispatch)
        t.join(timeout=1.0)
