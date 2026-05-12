from typing import Optional, List
import time
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
import langchain_openai.chat_models.base as base_module
from langchain_core.messages import AIMessage
from .config import load_config
import enum

# Monkey patch to support Google Vertex AI thought_signature
_original_convert_dict_to_message = base_module._convert_dict_to_message
_original_lc_tool_call_to_openai_tool_call = base_module._lc_tool_call_to_openai_tool_call

def _patched_convert_dict_to_message(_dict):
    msg = _original_convert_dict_to_message(_dict)
    if isinstance(msg, AIMessage) and msg.tool_calls:
        raw_tool_calls = _dict.get("tool_calls", [])
        # Assuming the order is preserved (it should be)
        if len(msg.tool_calls) == len(raw_tool_calls):
            for tc, raw_tc in zip(msg.tool_calls, raw_tool_calls):
                if "signature" in raw_tc:
                    tc["signature"] = raw_tc["signature"]
    return msg

def _patched_lc_tool_call_to_openai_tool_call(tool_call):
    d = _original_lc_tool_call_to_openai_tool_call(tool_call)
    if "signature" in tool_call:
        d["signature"] = tool_call["signature"]
    return d


class TokenTracker:
    """Track token usage per LLM role (agent vs subagent)."""
    def __init__(self):
        self._stats = {}  # {label: {"input_tokens": int, "output_tokens": int, "calls": int}}

    def record(self, label: str, usage_metadata: dict):
        if not usage_metadata:
            return
        if label not in self._stats:
            self._stats[label] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
        s = self._stats[label]
        s["input_tokens"] += usage_metadata.get("input_tokens", 0)
        s["output_tokens"] += usage_metadata.get("output_tokens", 0)
        s["calls"] += 1

    def summary(self) -> dict:
        total_in = sum(s["input_tokens"] for s in self._stats.values())
        total_out = sum(s["output_tokens"] for s in self._stats.values())
        return {
            "per_role": dict(self._stats),
            "total": {"input_tokens": total_in, "output_tokens": total_out,
                      "total_tokens": total_in + total_out},
        }

    def print_summary(self):
        s = self.summary()
        print("" + "=" * 60)
        print("TOKEN USAGE SUMMARY")
        print("=" * 60)
        for role, stats in s["per_role"].items():
            total = stats["input_tokens"] + stats["output_tokens"]
            print(f"  {role:20s}  calls={stats['calls']}  in={stats['input_tokens']}  out={stats['output_tokens']}  total={total}")
        t = s["total"]
        print("-" * 60)
        print(f"  {'TOTAL':20s}  in={t['input_tokens']}  out={t['output_tokens']}  total={t['total_tokens']}")
        print("=" * 60 + "")

    def save_to_log(self, log_dir: str):
        """Append token usage summary to the session log file."""
        import json as _json
        log_file = os.path.join(log_dir, "token_usage.json")
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "w", encoding="utf-8") as f:
            _json.dump(self.summary(), f, indent=2, ensure_ascii=False)


token_tracker = TokenTracker()


class LLM_TYPE(enum.Enum):
    AGENT = "model"
    SUBAGENT = "subagent_model"
    BASEAGENT = "base_model"


class RetryLLMWrapper:
    def __init__(self, llm, retries: int = 5, label: str = "unknown"):
        self.llm = llm
        self.retries = retries
        self.label = label

    def invoke(self, *args, **kwargs):
        attempts = 0
        while True:
            res = self.llm.with_retry(stop_after_attempt=self.retries).invoke(*args, **kwargs)
            
            # Check if choices is empty in response_metadata
            choices = getattr(res, "response_metadata", {}).get("choices")
            if choices is not None and len(choices) == 0:
                attempts += 1
                if attempts < self.retries:
                    time.sleep(0.5)
                    continue
            # Track token usage
            usage = getattr(res, "usage_metadata", None)
            if usage:
                token_tracker.record(self.label, usage)
            return res

    def bind_tools(self, *args, **kwargs):
        return RetryLLMWrapper(self.llm.bind_tools(*args, **kwargs), self.retries, self.label)

    def __getattr__(self, name):
        return getattr(self.llm, name)


def get_llm(type: LLM_TYPE = LLM_TYPE.AGENT) -> Optional[ChatOpenAI | AzureChatOpenAI]:
    cfg = load_config()
    llm = None
    if cfg.get("provider") == "bd":
        openai_cfg = cfg.get("bd") or {}

        base_url = openai_cfg.get("base_url") or "https://ark-cn-beijing.bytedance.net/api/v3"
        api_key = openai_cfg.get("api_key") or ""
        model = openai_cfg.get(type.value) or "gpt-4o-mini"

        temperature = float(openai_cfg.get("temperature") or 0.0)
        max_tokens = int(openai_cfg.get("max_tokens") or 8192)
    
        reasoning_effort = openai_cfg.get("reasoning_effort") or "minimal"
        
        llm = ChatOpenAI(base_url=base_url, api_key=api_key, model=model, temperature=temperature, max_tokens=max_tokens, reasoning_effort=reasoning_effort)
    else:
        google_cfg = cfg.get(f"{cfg.get('provider')}") or {}

        azure_endpoint = google_cfg.get("azure_endpoint") or "https://genai-sg-og.tiktok-row.org/gpt/openapi/online/v2/crawl"
        api_key = google_cfg.get("api_key") or ""
        api_version = google_cfg.get("api_version") or "2024-03-01-preview"
        model = google_cfg.get(type.value) or "gemini-3-pro-preview-new"

        temperature = float(google_cfg.get("temperature") or 0.0)
        max_tokens = int(google_cfg.get("max_tokens") or 8192)

        reasoning_effort = google_cfg.get("reasoning_effort") or "minimal"

        base_module._convert_dict_to_message = _patched_convert_dict_to_message
        base_module._lc_tool_call_to_openai_tool_call = _patched_lc_tool_call_to_openai_tool_call

        llm = AzureChatOpenAI(openai_api_type="azure", azure_endpoint=azure_endpoint, openai_api_key=api_key, deployment_name=model, temperature=temperature, max_tokens=max_tokens, openai_api_version=api_version, reasoning_effort=reasoning_effort) 
    
    proxies = {}
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    
    if llm:
        llm.openai_proxy = proxies
        return RetryLLMWrapper(llm, retries=6, label=type.value)
    return None 


agent_llm = get_llm()
subagent_llm = get_llm(LLM_TYPE.SUBAGENT)
