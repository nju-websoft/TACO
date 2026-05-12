import threading
import queue
import sys
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent.graph import app
from agent.invoke import SYSTEM_PROMPT
from agent.dispatch import global_dispatcher
from agent.context import compress_context
from agent.model import token_tracker
from agent.utils import get_root_dir, _SESSION_TIMESTAMP

def chat(instruction: str, dataset_dir: str = None):
    """
    Calls the agent system with a natural language instruction.
    Streams the output (thoughts, tool calls, final answer) to stdout.
    """
    if dataset_dir:
        from agent.components.planner import set_dataset_dir
        set_dataset_dir(dataset_dir)

    print(f"User: {instruction}\n")
    print("-" * 50)
    
    # Setup messages
    msgs = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=instruction)
    ]
    init = {"messages": msgs}

    # Queue to merge events
    q = queue.Queue()

    # Handler for global dispatcher (tool calls from deep within)
    def on_dispatch(etype, data):
        q.put((etype, data))

    global_dispatcher.register(on_dispatch)

    # Run app in separate thread
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
                etype, data = q.get(timeout=0.1)
            except queue.Empty:
                if not t.is_alive() and q.empty():
                    break
                continue

            if etype == "done":
                break
            if etype == "error":
                print(f"\n[Error] {data}")
                break

            if etype == "tool_call":
                name = data.get("name")
                args = data.get("args") or {}
                agent = data.get("agent")
                prefix = f"[{agent}] " if agent else ""

                # Format progress events with progress bar
                if "current" in args and "total" in args:
                    cur, tot = args["current"], args["total"]
                    pct = args.get("percent", 0)
                    filled = int(20 * pct / 100)
                    bar = chr(9608) * filled + chr(9617) * (20 - filled)
                    extra = " ".join(f"{k}={v}" for k, v in args.items() if k not in ("current", "total", "percent"))
                    extra_s = f" | {extra}" if extra else ""
                    print(f"\r> {prefix}{name} [{bar}] {cur}/{tot} ({pct}%){extra_s}", end="", flush=True)
                    if cur >= tot:
                        print()
                elif name.endswith("_done"):
                    detail = " | ".join(f"{k}={v}" for k, v in args.items())
                    print(f"\n> {prefix}{name}: {detail}")
                elif name.endswith("_reject"):
                    print(f"\n> {prefix}{name}: tool={args.get('tool')} reason={args.get('reason')}")
                else:
                    detail = ", ".join(f"{k}={v}" for k, v in args.items()) if args else ""
                    print(f"\n> {prefix}{name}({detail})")

            elif etype == "app_event":
                # State update from LangGraph
                event = data
                if not isinstance(event, dict):
                    continue
                
                for node_name, state_update in event.items():
                    if not isinstance(state_update, dict):
                        continue
                    
                    # 1. Check for tool requests (Executor deciding to call a tool)
                    tr = state_update.get("tool_request")
                    if tr:
                        name = tr.get("name")
                        args = tr.get("args")
                        print(f"\n> [Executor] Requesting Tool: {name}({args})")

                    # 2. Check for todos (Planner updates)
                    todos = state_update.get("todos")
                    if isinstance(todos, list) and todos:
                        print("\n> [Planner] Updated Todo List:")
                        for it in todos:
                            print(f"  - [{it.get('status')}] {it.get('name')}: {it.get('description')}")

                    # 3. Check for messages (AI output)
                    messages_list = state_update.get("messages")
                    if messages_list:
                        last_msg = messages_list[-1]
                        if isinstance(last_msg, AIMessage):
                            content = getattr(last_msg, "content", "")
                            # Avoid printing tool calls if they are part of content (OpenAI style)
                            if not last_msg.tool_calls and content:
                                print(f"\n[Agent]: {content}")
    finally:
        global_dispatcher.unregister(on_dispatch)
        t.join(timeout=1.0)
    
    token_tracker.print_summary()
    import os as _os
    token_tracker.save_to_log(_os.path.join(get_root_dir(), "log", _SESSION_TIMESTAMP))
    print("-" * 50)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        instruction = sys.argv[1]
    else:
        instruction = "Help me analyze the dataset in dataset/alpaca"
    chat(instruction)
