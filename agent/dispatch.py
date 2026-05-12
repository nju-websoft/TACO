from typing import Callable, List
import threading

class EventDispatcher:
    def __init__(self):
        self._handlers: List[Callable[[str, dict], None]] = []
        self._lock = threading.Lock()

    def register(self, handler: Callable[[str, dict], None]):
        with self._lock:
            self._handlers.append(handler)

    def unregister(self, handler: Callable[[str, dict], None]):
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def emit_progress(self, name: str, current: int, total: int, agent: str = None, **extra):
        """Convenience wrapper for progress-style events."""
        args = {"current": current, "total": total, "percent": round(current / total * 100) if total else 0}
        args.update(extra)
        self.emit_tool_call(name=name, args=args, agent=agent)

    def emit_tool_call(self, name: str, args: dict, agent: str = None):
        with self._lock:
            # Copy handlers to avoid issues if handlers change during iteration
            handlers = list(self._handlers)
        
        for h in handlers:
            h("tool_call", {"name": name, "args": args, "agent": agent})

global_dispatcher = EventDispatcher()
