import os
import sys
import json
import numpy as np
import shutil
import glob
import faiss
import pickle
import uuid
import torch
from sentence_transformers import SentenceTransformer
# Add project root to path to allow importing agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.model import get_llm
from agent.config import load_config
from agent.tools.python import python_run
from agent.tools.bash import bash_exec
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from datetime import datetime
from finetune.judger_prompt import JUDGER_SYS_PROMPT
from transformers import AutoTokenizer
from agent.utils import safe_truncate, get_clean_content


def get_remaining_id2data(original_data_dir, except_data_dir):
    original_id2data_dict = get_id2data(original_data_dir)
    except_id2data_dict = get_id2data(except_data_dir)
    remaining_id2data_dict = {k: v for k, v in original_id2data_dict.items() if k not in except_id2data_dict.keys()}
    return remaining_id2data_dict


def get_id2data(data_dir):
    id2data_dict = {}
    all_data_files = glob.glob(os.path.join(data_dir, "*.json"))
    for file_path in all_data_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if 'id' in item:
                    data_id = item['id']
                else:
                    print(f"[Warning] No id field in {file_path}. Assigning random id: {data_id}")
                    data_id = str(uuid.uuid4())[:8]
                id2data_dict[data_id] = item
    return id2data_dict


@tool
def output_answer(answer: str):
    """Output the final report if the task has been completed."""
    return answer


class Judger:
    def __init__(self, log_dirs, data_dir, original_data_dir, max_turns:int=24, top_k_data:int=5, add_percent:float=0.02, plateau_data_limits:int=3):
        """
        Args:
            log_dirs: Single string path or list of string paths.
        """
        self.log_dirs = [log_dirs] if isinstance(log_dirs, str) else log_dirs
            
        self.llm = get_llm()

        self.data_dir = data_dir
        self.id2data_dict = get_id2data(data_dir)
        self.remaining_id2data_dict = get_remaining_id2data(original_data_dir, data_dir)

        self.max_turns = max_turns
        self.top_k_data = top_k_data
        
        combined_data = {
            "train": [],
            "val": []
        }
        
        for d in self.log_dirs:
            train_f = os.path.join(d, "train_loss_ids.jsonl")
            val_f = os.path.join(d, "val_loss_ids.jsonl")

            print(f"[Judger] Processing {train_f} and {val_f}")
            
            combined_data["train"].extend(self.loss_log2list(train_f))
            combined_data["val"].extend(self.loss_log2list(val_f))
                
        print(f"[Judger] Collected {len(combined_data['train'])} training records and {len(combined_data['val'])} validation records.")
        
        if not combined_data["train"] or not combined_data["val"]:
            print("[Judger] No data found to analyze.")
            return
        self.combined_data = combined_data

        self.faiss_path = next(iter(glob.glob(os.path.join(original_data_dir, "*.index"))))
        self.idmapping_path = next(iter(glob.glob(os.path.join(original_data_dir, "*.pkl"))))
        
        self.faiss_index = faiss.read_index(self.faiss_path)
        with open(self.idmapping_path, "rb") as f:
            self.faiss_idmapping = pickle.load(f)

        self.add_percent = add_percent
        
        cfg = load_config()
        bd_cfg = cfg.get("bd", {})
        
        self.embed_model_path = bd_cfg.get("embed_model_path", "Qwen/Qwen3-Embedding-0.6B")
        print(f"[Judger] Loading local SentenceTransformer model from: {self.embed_model_path}")
        
        try:
            self.embed_model = SentenceTransformer(self.embed_model_path, trust_remote_code=True)
            if torch.cuda.is_available():
                device = torch.device(f"cuda:{torch.cuda.current_device()}")
                self.embed_model = self.embed_model.to(device)
                print(f"[Judger] Embedding model moved to {device}.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model_path)
        except Exception as e:
            print(f"[Judger] Error loading embedding model: {e}")
            self.embed_model = None

        self.plateau_data_limits = plateau_data_limits


    def loss_log2list(self, log_file):
        total_loss_list = []
        if not os.path.exists(log_file):
            return total_loss_list
            
        with open(log_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    loss_list = [x['loss'] for x in record.get('data', [])]
                    avg_loss = record.get('avg_loss', 0.0)
                    std_loss = np.std(loss_list).item() if loss_list else 0.0
                    epoch = record.get('epoch', -1)
                    total_loss_list.append({
                        "epoch": epoch,
                        "avg_loss": avg_loss,
                        "std_loss": std_loss,
                        "loss_list": loss_list,
                        "file": log_file
                    })
                except json.JSONDecodeError:
                    continue
        
        return total_loss_list
        

    def analyze(self):
        """
        Analyzes the loss logs using the LLM agent.
        """
        # Dump to temp files
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".judger_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        train_dump_path = os.path.join(tmp_dir, "train_loss.json")
        val_dump_path = os.path.join(tmp_dir, "val_loss.json")
        
        with open(train_dump_path, "w", encoding="utf-8") as f:
            json.dump(self.combined_data["train"], f, indent=2)
        with open(val_dump_path, "w", encoding="utf-8") as f:
            json.dump(self.combined_data["val"], f, indent=2)
            
        print(f"[Judger] Dumped loss data to {train_dump_path} and {val_dump_path}")

        # Construct Prompt
        initial_msg = f"""I have aggregated the loss data into two JSON files:
1. Train Loss: {train_dump_path}
2. Validation Loss: {val_dump_path}

The file structure is a list of dicts, where each dict represents a record from the logs:
{{
  "epoch": int, 
  "avg_loss": float, 
  "std_loss": float,
  "loss_list": [float, ...], # List of losses for each step in this epoch
  "file": str
}}

Please analyze both the training and validation loss trends to detect any anomalies (spikes, plateaus, overfitting, etc.).
You should:
1. Create a file with bash_exec tool and write Python script first to load these files.
2. Calculate statistics or check for specific patterns by python_run tool.
3. Report your findings as the structure above through output_answer tool.

**ATTENTION**
- Only add scripts at the {tmp_dir} directory, and output the results of scripts executing in this directory.  Remember to THINK before you ACT.
- DO NOT analyze the training loss, and do not use function that detecting spike or plateau on training loss.
- NEVER call more than one tool per turn. If you want to call tool, show them in the tool_calls part instead of output them in content part. DO THE TASK STEP BY STEP.
"""
        # Agent Loop
        tools = [python_run, bash_exec, output_answer]
        llm_with_tools = self.llm.bind_tools(tools, tool_choice="any")
        
        messages = [
            SystemMessage(content=JUDGER_SYS_PROMPT),
            HumanMessage(content=initial_msg)
        ]
        
        print("[Judger] Invoking Agent...")
        
        # Max turns for tool usage
        final_response = ""
        for i in range(self.max_turns):
            print(f"\n[Judger] --- Turn {i+1}/{self.max_turns} ---")
            try:
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                # Log Thought (Model Output)
                if response.content:
                    print(f"[Judger] Model Thought:\n{response.content}\n")
                
                print(f"[Judger] Agent requested tool(s):")
                for tc in response.tool_calls:
                     print(f"  - Tool: {tc['name']}")
                     print(f"  - Args: {tc['args']}...")
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]
                    
                    tool_output = f"Error: Tool {tool_name} not found."
                    
                    try:
                        if "python" in tool_name:
                            tool_output = python_run.invoke(tool_args)
                        elif "bash" in tool_name:
                            tool_output = bash_exec.invoke(tool_args)
                        elif "output_answer" in tool_name:
                            tool_output = output_answer.invoke(tool_args)
                            final_response = tool_output
                            break
                    except Exception as e:
                        tool_output = f"Error executing {tool_name}: {str(e)}"
                    
                    # Log Tool Output (Truncated)
                    tool_output_str = str(tool_output)
                    print(f"[Judger] Tool Output ({tool_name}):\n{tool_output_str[:200]}\n")
                        
                    messages.append(ToolMessage(tool_call_id=tool_id, content=tool_output_str, name=tool_name))
                
                if len(final_response) > 0:
                    break
            except Exception as e:
                print(f"[Judger] Error during agent execution: {e}")
                break  

        print("\n" + "="*30)
        print("JUDGER REPORT")
        print("="*30)
        print(final_response)
        print("="*30 + "\n")

        bad_loss_list = []
        try:
            bad_loss_list = get_clean_content(final_response)
        except Exception as e:
            print(f"[Judger] Failed to parse JSON: {e}")

        self.bad_loss_list = bad_loss_list
        self.handle_spike()
        self.handle_plateau()

        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception as e:
            print(f"[Judger] Warning: Failed to remove tmp_dir {tmp_dir}: {e}")

        # Free GPU memory: bge-m3 (~10GB) and FAISS index were loaded in
        # __init__ on the rank running this Judger (typically rank 0). Without
        # an explicit del + empty_cache, PyTorch's CUDA caching allocator may
        # hold the memory long enough to OOM the next epoch's training step
        # (compounding with DeepSpeed activations / GatheredParameters in test
        # eval) — and the OOM would manifest only on the rank running Judger,
        # producing another asymmetric collective failure.
        try:
            if hasattr(self, "embed_model") and self.embed_model is not None:
                del self.embed_model
                self.embed_model = None
            if hasattr(self, "faiss_index"):
                del self.faiss_index
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Judger] Warning: failed to release embed_model/faiss: {e}")


    def handle_plateau(self):
        enriched_bad_loss_list = []

        for anomaly in self.bad_loss_list:
            if anomaly.get("type") != "plateau":
                continue
            if not anomaly.get("epoch_range"):
                print(f"[Judger] Invalid anomaly: {anomaly} with no epoch_range")
                continue
            
            epoch_range = anomaly.get("epoch_range")
            phase = anomaly.get("phase")
            if phase != "val":
                print(f"[Warning] Only process validation loss. Phase: {phase}")
            
            target_data = self.combined_data.get(phase, [])
            start_epoch, end_epoch = epoch_range.split("-")
            start_epoch = int(start_epoch)
            end_epoch = int(end_epoch)
            if start_epoch < 0 or end_epoch < 0 or start_epoch > end_epoch:
                print(f"[Warning] Invalid epoch_range: {epoch_range}")
                continue

            for epoch in range(start_epoch, end_epoch + 1):
                target_record = next((r for r in target_data if r["epoch"] == epoch), None)
                if not target_record:
                    print(f"[Warning] Invalid record: {target_record}")
                    continue
               
                tgt_record_file = target_record.get("file", "")
                if not tgt_record_file:
                    print(f"[Warning] Invalid record: {target_record}")
                    continue

                with open(tgt_record_file, "r", encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if record.get("epoch") == epoch:
                                data_batch = record.get("data", [])
                                sorted_data_batch = sorted(data_batch, key=lambda x: x["loss"])
                                for i in range(self.plateau_data_limits):
                                    enriched_bad_loss_list.append({"type": anomaly.get("type"), "loss2data": sorted_data_batch[i]})
                                break
                        except json.JSONDecodeError:
                            print(f"[Warning] Invalid line in {tgt_record_file}: {line.strip()}")
                            continue
        
        if len(enriched_bad_loss_list) == 0:
            print(f"[Judger] No plateau anomaly found.")
            return

        data_ids_freq = {}
        for ele in enriched_bad_loss_list:
            data_ids = ele["loss2data"].get("ids", [])
            if not data_ids:
                print(f"[Judger] Empty ids in {ele}")
                continue
            for data_id in data_ids:
                if data_id in data_ids_freq:
                    data_ids_freq[data_id] += 1
                else:
                    data_ids_freq[data_id] = 1

        sorted_by_datafreq = sorted(data_ids_freq.items(), key=lambda x: x[1], reverse=True)
        high_freq_data = []
        # Get top k data, but ensure we don't exceed available data
        count = min(self.top_k_data, len(sorted_by_datafreq))
        for i in range(count):
            data_id = sorted_by_datafreq[i][0]
            if data_id in self.id2data_dict:
                data_text = self.id2data_dict[data_id]
                high_freq_data.append(data_text)
        
        rewritten_data_list = []
        for item in high_freq_data:
            rewritten_item = self.rewrite_data(item)
            if rewritten_item:
                 rewritten_data_list.append(rewritten_item)
        
        # Add data to remaining data
        candidates_ids = self.add_data(rewritten_data_list)
        candidates_data = []
        for cid in candidates_ids:
            if cid in self.remaining_id2data_dict:
                candidates_data.append({
                    "id": cid,
                    "instruction": self.remaining_id2data_dict[cid].get('instruction', ''),
                    "input": self.remaining_id2data_dict[cid].get('input', ''),
                    "output": self.remaining_id2data_dict[cid].get('output', ''),
                    "text": self.remaining_id2data_dict[cid].get('text', ''),
                })

        added_file_name = f"plateau_data_{epoch_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.data_dir, added_file_name), "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, indent=2)

        print(f"[Judger] Add {len(candidates_data)} data items to {added_file_name}")


    def rewrite_data(self, data_item):
        """
        Rewrite the data item to increase difficulty using LLM.
        """
        prompt = f"""You are an expert in creating challenging SFT (Supervised Fine-Tuning) datasets.
Below is a data item used for training a model. Your task is to rewrite this data item to make it significantly more difficult and challenging for the model to learn, while maintaining the same JSON structure and field names.

Original Data Item:
{json.dumps(data_item, indent=2, ensure_ascii=False)}

Requirements:
1. Increase the complexity of the instruction or input (if present).
2. Ensure the output logic remains consistent but requires more reasoning or detailed explanation.
3. The output MUST be a valid JSON object with the exact same keys as the original.
4. Do NOT wrap the output in markdown code blocks. Just return the raw JSON string.
5. If find 'id' in original data item, just randomly generate a new id for rewritten data item.

Rewrite:
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # Attempt to clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            rewritten_item = json.loads(content)
            
            # Ensure keys match
            if set(rewritten_item.keys()) == set(data_item.keys()):
                 # Assign a new ID if present
                if 'id' in rewritten_item:
                    rewritten_item['id'] = str(uuid.uuid4())[:8]
                return rewritten_item
            else:
                print(f"[Judger] Warning: Rewritten data keys do not match original. Skipping.")
                return None
                
        except json.JSONDecodeError:
            print(f"[Judger] Error decoding LLM response as JSON: {content[:100]}...")
            return None
        except Exception as e:
            print(f"[Judger] Error during data rewriting: {e}")
            return None


    def handle_spike(self):
        # Post-process bad loss list to add raw text
        enriched_bad_loss_list = []
        
        for anomaly in self.bad_loss_list:
            if anomaly.get("type") != "spike":
                continue
            if anomaly.get("step_index", -1) == -1:
                print(f"[Judger] Invalid anomaly: {anomaly} with no step_index")
                continue
            
            epoch = anomaly.get("epoch")
            step_index = anomaly.get("step_index")
            phase = anomaly.get("phase")
            if phase != "val":
                print(f"[Warning] Only process validation loss. Phase: {phase}")

            target_data = self.combined_data.get(phase, [])
            # Find the record for this epoch
            target_record = next((r for r in target_data if r["epoch"] == epoch), None)
            tgt_record_file = target_record.get("file", "")
            if not tgt_record_file:
                print(f"[Judger] Invalid record: {target_record}")
                continue
            
            with open(tgt_record_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if record.get("epoch") == epoch:
                            data_batch = record.get('data', [])
                            if 0 <= step_index < len(data_batch):
                                enriched_bad_loss_list.append({"type": anomaly.get("type"), "loss2data": data_batch[step_index]})
                            break
                    except json.JSONDecodeError:
                        print(f"[Warning] Invalid line in {tgt_record_file}: {line.strip()}")
                        continue
        
        if len(enriched_bad_loss_list) == 0:
            print(f"[Judger] No spike anomaly found.")
            return
        
        data_ids_freq = {}
        for ele in enriched_bad_loss_list:
            data_ids = ele["loss2data"].get("ids", [])
            if not data_ids:
                print(f"[Judger] Empty data_ids in anomaly: {ele}")
                continue
            for data_id in data_ids:
                if data_id in data_ids_freq:
                    data_ids_freq[data_id] += 1
                else:
                    data_ids_freq[data_id] = 1
        
        sorted_by_datafreq = sorted(data_ids_freq.items(), key=lambda x: x[1], reverse=True)
        high_freq_data = []
        # Get top k data, but ensure we don't exceed available data
        count = min(self.top_k_data, len(sorted_by_datafreq))
        for i in range(count):
            data_id = sorted_by_datafreq[i][0]
            if data_id in self.id2data_dict:
                data_text = self.id2data_dict[data_id]
                high_freq_data.append(data_text)
        
        if not high_freq_data:
            print("[Judger] No high frequency data found from anomalies.")
            return []
        
        candidates_ids = self.add_data(high_freq_data)
        candidates_data = []
        for cid in candidates_ids:
            if cid in self.remaining_id2data_dict:
                candidates_data.append({
                    "id": cid,
                    "instruction": self.remaining_id2data_dict[cid].get('instruction', ''),
                    "input": self.remaining_id2data_dict[cid].get('input', ''),
                    "output": self.remaining_id2data_dict[cid].get('output', ''),
                    "text": self.remaining_id2data_dict[cid].get('text', ''),
                })
        
        added_file_name = f"spike_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.data_dir, added_file_name), "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, indent=2)
        
        print(f"[Judger] Added {len(candidates_data)} spike data items to {added_file_name}")


    def filter_candidates_with_llm(self, query_data, candidate_ids, batch_size=20):
        selected_ids = []
        
        query_text = query_data.get('text', '')
        if not query_text:
             query_text = f"Instruction: {query_data.get('instruction', '')}\nInput: {query_data.get('input', '')}\nOutput: {query_data.get('output', '')}"
        
        candidates_data = []
        for cid in candidate_ids:
            if cid in self.remaining_id2data_dict:
                candidates_data.append((cid, self.remaining_id2data_dict[cid]))
        
        for i in range(0, len(candidates_data), batch_size):
            batch = candidates_data[i:i+batch_size] if i+batch_size <= len(candidates_data) else candidates_data[i:]
        
            candidates_str = ""
            candidates_dict = {}
            for idx, (cid, data) in enumerate(batch):
                text = data.get('text', '')
                if not text:
                     text = f"Instruction: {data.get('instruction', '')}\nInput: {data.get('input', '')}\nOutput: {data.get('output', '')}"
                candidates_str += f"Index: {idx}\nText: {text}...\n\n"
                candidates_dict[idx] = cid
            
            prompt = f"""Given a Query data item, select the indices of the Candidate data items that are semantically relevant and helpful for training a model on similar tasks.
            
Query:
{query_text}

Candidates:
{candidates_str}

ONLY Return a int list of integers representing the indices of relevant candidates. e.g. [0, 2]. If none, just return [].
"""
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                content = response.content
                # Parse JSON
                import re
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                print(f"[Judger] LLM Response: {content}")
                if json_match:
                    indices = json.loads(json_match.group())
                    for idx in indices:
                        if isinstance(idx, int) and 0 <= idx < len(candidates_dict):
                            selected_ids.append(candidates_dict[idx])
            except Exception as e:
                print(f"[Judger] LLM Filter Error: {e}")
                
        return selected_ids


    def add_data(self, high_freq_data):
        origin_data_len = len(self.faiss_idmapping)
        num_high_freq = len(high_freq_data)
        all_candidate_ids = []
        
        # Calculate target count per spike item
        target_count = int((origin_data_len * self.add_percent) / num_high_freq)
        if target_count <= 0:
            target_count = 1
        
        print(f"[Judger] Origin data len: {origin_data_len}")
        print(f"[Judger] Target valid neighbors per item: {target_count}")

        for tgt_data in high_freq_data:
            text = tgt_data.get('text', '')
            if not text:
                # Try constructing text if missing
                print(f"[Judger] No text in data: {tgt_data}")
                text = f"{tgt_data.get('instruction', '')}\n{tgt_data.get('input', '')}\n{tgt_data.get('output', '')}"
            
            if not text.strip():
                continue
            
            potential_ids = []
            try:
                if self.embed_model:
                    # 使用本地 SentenceTransformer 模型
                    # model.encode returns numpy array
                    embedding = self.embed_model.encode([safe_truncate(self.tokenizer, text)], convert_to_numpy=True)[0]
                    emb_np = np.array([embedding], dtype='float32')
                    faiss.normalize_L2(emb_np)
                    
                    # Iterative search
                    current_k = target_count
                    
                    while True:
                        # Limit k
                        search_k = min(current_k, origin_data_len)
                        D, I = self.faiss_index.search(emb_np, search_k)
                        
                        # Find new potential candidates
                        for idx in I[0]:
                            if idx != -1 and idx < len(self.faiss_idmapping):
                                data_id = self.faiss_idmapping[idx]
                                if data_id not in self.id2data_dict and data_id in self.remaining_id2data_dict:
                                    potential_ids.append(data_id)
                        
                        if len(potential_ids) >= target_count:
                            potential_ids = potential_ids[:target_count]
                            break
                        
                        if search_k >= origin_data_len:
                            print(f"[Judger] Warning: Only found {len(potential_ids)} valid candidates for data item (target: {target_count})")
                            break
                            
                        current_k += target_count
                        
            except Exception as e:
                print(f"[Judger] Embedding/Search error: {e}")

            print(f"[Judger] Potential candidates number: {len(potential_ids)}")
            final_valid_ids = self.filter_candidates_with_llm(tgt_data, potential_ids)
            print(f"[Judger] Final valid candidates number: {len(final_valid_ids)}")

            all_candidate_ids.extend(final_valid_ids)

        return all_candidate_ids



if __name__ == "__main__":
    # Test
    base_dir = "finetune/log/20260129_192608/"
    data_dir = "dataset/alpaca"
    test_dir = [base_dir + f"{3*i}_train_{i}_val" for i in range(1, 10)]
    j = Judger(test_dir, data_dir)
    j.analyze()
