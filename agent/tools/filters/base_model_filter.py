from concurrent.futures import as_completed
import numpy as np
import torch
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn import CrossEntropyLoss
from sentence_transformers import util, SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from agent.config import load_config
from agent.tools.filters.rough_filter import _load_json_or_jsonl
from agent.dispatch import global_dispatcher
from agent.utils import safe_truncate, time_logger



class BaseModelTools:
    def __init__(self, dataset_path: Optional[str] = None, chat_device=None, embed_device=None):
        cfg = load_config()
        openai_cfg = cfg.get(cfg.get("provider"), "bd")

        # Read device config, allow override via constructor args
        chat_device = chat_device or str(openai_cfg.get("chat_device"))
        embed_device = embed_device or int(openai_cfg.get("embed_device"))

        self.base_model_url = openai_cfg.get("base_model_url", None)
        self.base_api_model_name = openai_cfg.get("base_api_model_name", "")
        self.base_model_client = OpenAI(base_url=self.base_model_url, api_key=openai_cfg.get("api_key", ""))
        
        self.base_model_name = openai_cfg.get("base_model_path", "")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        bmf_cfg = load_config().get('base_model_filter')
        self.inner_bs = bmf_cfg.get('inner_metrics_bs')
        
        # Parse chat_device: supports single ("2") or multi-gpu ("2,3")
        chat_gpus = [int(x.strip()) for x in str(chat_device).split(",")]
        self._chat_gpus = chat_gpus
        if len(chat_gpus) == 1:
            self._chat_device_map = {"": f"cuda:{chat_gpus[0]}"}
            self._chat_max_memory = None
        else:
            self._chat_device_map = "auto"
            # Query actual GPU memory instead of hardcoding; reserve 1 GiB headroom
            self._chat_max_memory = {}
            for g in chat_gpus:
                total = torch.cuda.get_device_properties(g).total_memory / (1024 ** 3)
                self._chat_max_memory[g] = f"{int(total - 1)}GiB"
            # Explicitly block all non-target GPUs so device_map="auto" stays isolated
            for g in range(torch.cuda.device_count()):
                if g not in chat_gpus:
                    self._chat_max_memory[g] = "0GiB"
            self._chat_max_memory["cpu"] = "0GiB"
            print(f"[BaseModel] max_memory: {self._chat_max_memory}")
        
        print(f"[BaseModel] Loading Chat Model: {self.base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16, 
            device_map=self._chat_device_map,
            max_memory=self._chat_max_memory,
            trust_remote_code=True
        )
        self.model.eval()

        self.embed_model_path = openai_cfg.get("embed_model_path", "Qwen/Qwen3-Embedding-0.6B")
        self.embed_model_tokenizer = AutoTokenizer.from_pretrained(self.embed_model_path, trust_remote_code=True)
        print(f"[BaseModel] Loading Embedding Model: {self.embed_model_path}")
        self.embed_model = SentenceTransformer(self.embed_model_path, device=f"cuda:{embed_device}")

        self.embed_model_url = openai_cfg.get("embed_url", None)
        self.embed_api_model_name = openai_cfg.get("embed_api_model_name", "Qwen/Qwen3-Embedding-0.6B")
        self.embed_model_client = OpenAI(base_url=self.embed_model_url, api_key=openai_cfg.get("api_key", ""))

        self.dataset = []
        self.anchor = "### Response:\n"
        
        # Cache for metrics: {idx: {"nll": float, "entropy": float}}
        self.metrics_cache = {}

        if dataset_path:
            self.load_dataset(dataset_path)

    def load_dataset(self, dataset_path: str):
        self.dataset = _load_json_or_jsonl(dataset_path)
        self.metrics_cache = {} # Reset cache on new dataset load
        print(f"[BaseModel] Loaded {len(self.dataset)} samples from {dataset_path}")
        
    @time_logger
    def _batch_calculate_metrics(self, batch_items: List[Dict]) -> List[Dict]:
        self.tokenizer.padding_side = "right"

        full_texts = [item.get("text", "") for item in batch_items]
        
        # 检查并警告过长样本
        for idx, text in enumerate(full_texts):
            token_count = len(self.tokenizer.encode(text))
            if token_count > 4096:
                print(f"[BaseModel] Warning: Sample {idx} has {token_count} tokens, will be truncated")

        encodings = self.tokenizer(
            full_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=4096
        ).to(next(self.model.parameters()).device)
        
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        labels = input_ids.clone()

        # 优化 1: 尽量缩短 Labels 处理逻辑，减少中间变量
        for i, text in enumerate(full_texts):
            prompt_text = text.split(self.anchor)[0] + self.anchor
            prompt_len = len(self.tokenizer.encode(prompt_text))
            labels[i, :prompt_len] = -100
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
            
            # 优化 2: 立即删除不再需要的 inputs，释放几百 MB 空间
            del input_ids, attention_mask, encodings, outputs

            # 3. 计算 NLL
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            token_losses = token_losses.view(shift_labels.size())

            # 优化 3: 节省熵的计算内存 (使用 logsumexp 技巧或原地操作)
            # 不再保留 probs 矩阵，直接计算
            log_p = torch.log_softmax(shift_logits, dim=-1)
            token_entropies = -(torch.exp(log_p) * log_p).sum(dim=-1)
            
            # 优化 4: 计算完指标立即释放巨大的 logits
            del shift_logits, log_p, logits

        # 5. 聚合结果
        batch_results = []
        token_losses = token_losses.cpu() # 移到 CPU 处理，腾出显存
        token_entropies = token_entropies.cpu()
        shift_labels = shift_labels.cpu()

        for i in range(len(batch_items)):
            mask = (shift_labels[i] != -100)
            v_loss = token_losses[i][mask]
            v_ent = token_entropies[i][mask]
            
            batch_results.append({
                "nll": float(v_loss.mean().item()) if v_loss.numel() > 0 else 0.0,
                "entropy": float(v_ent.mean().item()) if v_ent.numel() > 0 else 0.0
            })
        
        # 优化 5: 强制清理
        del token_losses, token_entropies, shift_labels, labels
        torch.cuda.empty_cache()
        
        return batch_results
        
    @time_logger
    def get_metrics_in_batches(self, start_idx: int, end_idx: int):
        all_results = {"nll": [], "entropy": []}
        
        # Identify indices that need calculation
        indices_to_calc = []
        for i in range(start_idx, end_idx):
            if i not in self.metrics_cache:
                indices_to_calc.append(i)
        
        # Calculate in batches if there are missing metrics
        if indices_to_calc:
            subset_to_calc = [self.dataset[i] for i in indices_to_calc]
            
            for i in range(0, len(subset_to_calc), self.inner_bs):
                batch = subset_to_calc[i : i + self.inner_bs]
                current_batch_indices = indices_to_calc[i : i + self.inner_bs]
                global_dispatcher.emit_progress(name="compute_metrics_batch", current=min(i + self.inner_bs, len(indices_to_calc)), total=len(indices_to_calc), agent="base_model_filter")
                metrics = self._batch_calculate_metrics(batch)
                
                # 每10个batch强制清理显存
                if (i // self.inner_bs) % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for idx_in_batch, m in enumerate(metrics):
                    global_idx = current_batch_indices[idx_in_batch]
                    nll = m["nll"]
                    entropy = m["entropy"]
                    
                    self.metrics_cache[global_idx] = {
                        "nll": nll,
                        "entropy": entropy,
                    }

        for i in range(start_idx, end_idx):
            if i in self.metrics_cache:
                cached = self.metrics_cache[i]
                all_results["nll"].append(cached["nll"])
                all_results["entropy"].append(cached["entropy"])
            else:
                print(f"[BaseModel] Warning: No metrics cached for index {i}.")
                all_results["nll"].append(0.0)
                all_results["entropy"].append(0.0)
        
        # 强制清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        return all_results

    def get_loss_metrics(self, start_idx: int, end_idx: int) -> List[float]:
        global_dispatcher.emit_progress(name="compute_loss", current=end_idx, total=len(self.dataset), agent="base_model_filter")
        return self.get_metrics_in_batches(start_idx, end_idx)["nll"]

    def get_entropy_metrics(self, start_idx: int, end_idx: int) -> List[float]:
        global_dispatcher.emit_progress(name="compute_entropy", current=end_idx, total=len(self.dataset), agent="base_model_filter")
        return self.get_metrics_in_batches(start_idx, end_idx)["entropy"]
    
    def get_mean_diff_metrics(self, start_idx: int, end_idx: int, layer_type: str = None) -> List[float]:
        """Calculate mean parameter difference (ResoFilter metric) by reusing the chat model."""
        if layer_type is None:
            num_layers = self.model.config.num_hidden_layers
            layer_type = f"model.layers.{num_layers - 1}.mlp.up_proj.weight"

        global_dispatcher.emit_progress(name="compute_mean_diff", current=end_idx, total=len(self.dataset), agent="base_model_filter")

        batch_items = self.dataset[start_idx:end_idx]
        device = next(self.model.parameters()).device

        # Snapshot original weights (only the target layer to save memory)
        original_param = self.model.state_dict()[layer_type].clone()
        # Full state dict needed for restoration after each sample
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        self.model.train()
        mean_diffs = []
        for i, item in enumerate(batch_items):
            self.model.load_state_dict(original_state)
            optimizer.zero_grad()

            text = item.get("text", "")
            encoding = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
            outputs = self.model(**encoding, labels=encoding.input_ids)
            outputs.loss.backward()
            optimizer.step()

            diff = (self.model.state_dict()[layer_type] - original_param).abs().mean().item()
            mean_diffs.append(diff)

            del encoding, outputs
            if i % 20 == 0:
                torch.cuda.empty_cache()
                global_dispatcher.emit_progress(name="compute_mean_diff", current=start_idx + i + 1, total=len(self.dataset), agent="base_model_filter")

        # Restore model to original state and eval mode
        self.model.load_state_dict(original_state)
        self.model.eval()
        del original_state, original_param, optimizer
        torch.cuda.empty_cache()

        for i, diff in enumerate(mean_diffs):
            idx = start_idx + i
            if idx not in self.metrics_cache:
                self.metrics_cache[idx] = {}
            self.metrics_cache[idx]["mean_diff"] = diff

        return mean_diffs

    def normalize_metrics(self, metric_name: str):
        """Normalize metric values to [0, 1] range using min-max normalization"""
        values = [self.metrics_cache[idx].get(metric_name) for idx in self.metrics_cache if metric_name in self.metrics_cache[idx]]

        if not values or len(values) < 2:
            return

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            for idx in self.metrics_cache:
                if metric_name in self.metrics_cache[idx]:
                    self.metrics_cache[idx][metric_name] = 0.5
        else:
            for idx in self.metrics_cache:
                if metric_name in self.metrics_cache[idx]:
                    val = self.metrics_cache[idx][metric_name]
                    normalized = (val - min_val) / (max_val - min_val)
                    # Reverse for mean_diff: lower diff = higher quality
                    if metric_name == "mean_diff":
                        normalized = 1.0 - normalized
                    self.metrics_cache[idx][metric_name] = normalized

    def get_batch_generation(self, prompts: List[str], use_remote: bool = True) -> List[str]:
        if use_remote:
            indexed_prompts = list(enumerate(prompts))
            num_workers = min(len(prompts), 32)
            
            if num_workers == 0:
                print("[BaseModel] Error: No prompts provided for remote generation.")
                return []
            results = []
            def process_single(item):
                idx, prompt = item
                try:
                    resp = self.base_model_client.chat.completions.create(
                        model=self.base_api_model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1024,
                        temperature=0.0
                    )
                    text = resp.choices[0].message.content.strip()
                    return (idx, text)
                except Exception as e:
                    print(f"[BaseModel] Remote Error for prompt {idx}: {e}")
                    return (idx, "")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(process_single, item) for item in indexed_prompts]
                for f in futures:
                    results.append(f.result())
            
            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]

        try:
            # 必须设置左填充，否则生成会错乱
            with torch.no_grad():
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024
                ).to(next(self.model.parameters()).device)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False, # 对应 temperature=0.0
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
                # 只解码新生成的 Token
                input_len = inputs.input_ids.shape[1]
                generated_texts = [
                    self.tokenizer.decode(g[input_len:], skip_special_tokens=True) 
                    for g in output_ids
                ]
                return generated_texts
        except Exception as e:
            print(f"[BaseModel] Error in batch generation: {e}")
            return [""] * len(prompts)

    @time_logger  
    def get_remote_embedding(self, texts: List[str], use_remote: bool=True) -> np.ndarray:
        if not texts:
            print("[BaseModel] Error: No texts provided for embedding.")
            return None

        if not use_remote:
            return self.embed_model.encode(texts, convert_to_tensor=True)
        
        batch_size = 32
        all_embeddings = [None] * len(texts)
        num_workers = min(len(texts), 16)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                safe_batch = [safe_truncate(self.embed_model_tokenizer, t) for t in batch] 
                futures.append(executor.submit(self._call_embed_api, safe_batch, i))

            for f in as_completed(futures):
                start_idx, embeddings = f.result()
                for j, emb in enumerate(embeddings):
                    all_embeddings[start_idx + j] = emb

        return np.array(all_embeddings, dtype='float32')

    def _call_embed_api(self, texts, start_idx):
        """底层 API 调用"""
        try:
            response = self.embed_model_client.embeddings.create(
                input=texts,
                model=self.embed_api_model_name 
            )

            embs = [data.embedding for data in sorted(response.data, key=lambda x: x.index)]
            return start_idx, embs
        except Exception as e:
            print(f"Remote Embedding Error at {start_idx}: {e}")
            return start_idx, [[0.0] * 1024] * len(texts)

    @time_logger
    def get_drift_metrics(self, start_idx: int, end_idx: int) -> List[float]:
        batch_data = self.dataset[start_idx:end_idx]
        if not batch_data:
            return []
        
        global_dispatcher.emit_progress(name="compute_drift", current=end_idx, total=len(self.dataset), agent="base_model_filter")
        
        prompts = [
            f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n{self.anchor}" 
            for x in batch_data
        ]

        gen_resp = self.get_batch_generation(prompts, use_remote=True)
        ref_outputs = [x["output"] for x in batch_data]

        combined_texts = gen_resp + ref_outputs
        combined_embeddings = self.get_remote_embedding(combined_texts)

        if len(combined_embeddings) == 0:
            return [0.0] * len(batch_data)

        split_idx = len(gen_resp)
        gen_embeds = combined_embeddings[:split_idx]
        ref_embeds = combined_embeddings[split_idx:]

        def get_safe_cosine_drift(g, r):
            norm_g_val = np.linalg.norm(g)
            norm_r_val = np.linalg.norm(r)  
            if norm_g_val == 0 or norm_r_val == 0:
                print(f"[BaseModel] Warning: Zero norm for embedding {g} or {r}. Drift score set to 1.0.")
                return 1.0  
            norm_g = g / norm_g_val
            norm_r = r / norm_r_val
            return float(1.0 - np.dot(norm_g, norm_r))

        drift_scores = [get_safe_cosine_drift(g, r) for g, r in zip(gen_embeds, ref_embeds)]

        return drift_scores


if __name__ == "__main__":
    # Self-test logic
    print("Running self-test for BaseModelTools...")
    test_dataset_path = "dataset/trial_set/data.json"  # Ensure this path exists or use a dummy file
    
    # Create a dummy dataset if it doesn't exist for testing purposes
    import os

    try:
        tools = BaseModelTools(test_dataset_path, chat_device='1,2', embed_device=3)
        
        # Test NLL
        print("\nTesting get_loss_metrics...")
        nll_scores = tools.get_loss_metrics(0, 10)
        print(f"NLL Scores: {nll_scores}")
        
        
        # Test Entropy
        print("\nTesting get_entropy_metrics...")
        entropy_scores = tools.get_entropy_metrics(0, 10)
        print(f"Entropy Scores: {entropy_scores}")
        
        # Test Drift
        # print("\nTesting get_drift_metrics...")
        # drift_scores = tools.get_drift_metrics(0, 50)
        # print(f"Drift Scores: {drift_scores}")
        
        print("\nTesting get_mean_diff_metrics...")
        mean_diff_scores = tools.get_mean_diff_metrics(0, 10)
        print(f'Mean-Diff Scores: {mean_diff_scores}')
        print("\nSelf-test completed successfully.")
        
    except Exception as e:
        print(f"\nSelf-test failed: {e}")
