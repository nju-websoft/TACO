from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from typing import List, Dict, Any, Optional
from .config import limits
from .model import agent_llm
import re

COMPRESS_PROMPT = (
    "Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.\n"
    "This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.\n\n"
    "Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:\n\n"
    "1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:\n"
    "    - The user's explicit requests and intents\n"
    "    - Your approach to addressing the user's requests\n"
    "    - Key decisions, technical concepts and code patterns\n"
    "    - Specific details like:\n"
    "      - file names\n"
    "      - full code snippets\n"
    "      - function signatures\n"
    "    - Errors that you ran into and how you fixed them\n"
    "    - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.\n"
    "2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.\n\n"
    "Your summary should include the following sections:\n\n"
    "1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail\n"
    "2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.\n"
    "3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.\n"
    "4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.\n"
    "5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.\n"
    "6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.\n"
    "6. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.\n"
    "7. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.\n"
    "8. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.\n"
    "                        If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.\n\n"
    "<example>\n<analysis>\n[Your thought process, ensuring all points are covered thoroughly and accurately]\n</analysis>\n\n<summary>\n1. Primary Request and Intent:\n   [Detailed description]\n\n2. Key Technical Concepts:\n   - [Concept 1]\n   - [Concept 2]\n   - [...]\n\n3. Files and Code Sections:\n   - [File Name 1]\n      - [Summary of why this file is important]\n      - [Summary of the changes made to this file, if any]\n      - [Important Code Snippet]\n   - [File Name 2]\n      - [Important Code Snippet]\n   - [...]\n\n4. Errors and fixes:\n    - [Detailed description of error 1]:\n      - [How you fixed the error]\n      - [User feedback on the error if any]\n    - [...]\n\n5. Problem Solving:\n   [Description of solved problems and ongoing troubleshooting]\n\n6. All user messages:\n    - [Detailed non tool use user message]\n    - [...]\n\n7. Pending Tasks:\n   - [Task 1]\n   - [Task 2]\n   - [...]\n\n8. Current Work:\n   [Precise description of current work]\n\n9. Optional Next Step:\n   [Optional Next step to take]\n\n</summary>\n</example>\n\nAdditional Instructions:\nplease be short"
)

def compress_context(messages: List[object], max_context_chars: Optional[int] = None) -> List[object]:
    """Compress context if it exceeds limits by summarizing older messages."""
    if max_context_chars is None:
        max_context_chars = int(limits.get("max_context_chars", 30000))
        
    total_chars = sum(len(_get_msg_text(m)) for m in messages)
    
    if total_chars <= max_context_chars:
        return messages
        
    # Compression logic
    # We want to keep recent history intact as much as possible
    keep_recent_chars = min(max_context_chars // 2, limits.get("keep_recent_chars", 5000))
    recent_msgs = []
    current_chars = 0
    cutoff_idx = 0
    
    # Iterate backwards to find recent window
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        l = len(_get_msg_text(m))
        if current_chars + l > keep_recent_chars:
            cutoff_idx = i + 1
            break
        current_chars += l
        recent_msgs.insert(0, m)
        
    if cutoff_idx == 0:
        return messages
        
    msgs_to_summarize = messages[:cutoff_idx]
    if not msgs_to_summarize:
        return messages

    try:
        # Use the detailed prompt
        prompt_msg = HumanMessage(content=COMPRESS_PROMPT)
        # We pass the messages to be summarized + the prompt
        response = agent_llm.invoke(msgs_to_summarize + [prompt_msg])
        summary_text = getattr(response, "content", "") or ""
        
        # Parse XML if present
        match = re.search(r"<summary>(.*?)</summary>", summary_text, re.DOTALL)
        if match:
            summary_text = match.group(1).strip()
            
        # Create new context: Summary + Recent Messages
        return [SystemMessage(content=f"*** PREVIOUS CONVERSATION SUMMARY ***\n{summary_text}")] + recent_msgs
        
    except Exception as e:
        # If summarization fails, just return truncated recent messages
        return [SystemMessage(content="[Context compression failed]")] + recent_msgs

# Alias for backward compatibility
_compact_messages = compress_context

def _get_msg_text(m: object) -> str:
    if isinstance(m, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
        return getattr(m, "content", "") or ""
    return str(m)

def _to_dict_msg(m: object) -> Dict[str, Any]:
    if isinstance(m, HumanMessage):
        return {"role": "human", "content": getattr(m, "content", "") or ""}
    if isinstance(m, AIMessage):
        return {"role": "ai", "content": getattr(m, "content", "") or ""}
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": getattr(m, "content", "") or ""}
    if isinstance(m, ToolMessage):
        return {"role": "tool", "content": getattr(m, "content", "") or ""}
    return {"role": "human", "content": str(m)}
