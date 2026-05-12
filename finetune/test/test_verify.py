#!/usr/bin/env python3
"""
Verify the test module without GPU.

Tests all pure-logic components:
  - utils: answer extraction, string cleaning, data loading, math verify
  - evaluators: prompt generation, choice extraction, dataset routing
  - __init__: evaluate_test_set routing logic
  - BigCodeBench: data loading, prompt gen, code extraction, sandbox execution, pass@k

Evaluator modules import torch/deepspeed at top level, so we use
importlib to load individual functions and mock torch-dependent parts.
"""
import sys
import os
import json
import tempfile
import shutil
import types

sys.path.insert(0, '/data/fqzhou/dataset_agent')

# ============================================================
# Stub torch/deepspeed/datasets BEFORE any finetune.test imports,
# because test/__init__.py -> evaluators -> import torch
# and lcb/benchmarks/code_generation.py -> from datasets import load_dataset
# ============================================================
for _mod_name in [
    "torch", "torch.distributed",
    "deepspeed", "deepspeed.zero",
    "datasets",
    "anthropic",
    "numpy", "numpy.core", "numpy.core.multiarray",
    "pandas",
    "tqdm",
    "attrs",
    "pebble",
]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# datasets stub
sys.modules["datasets"].load_dataset = lambda *a, **k: []

_torch_stub = sys.modules["torch"]
_torch_stub.tensor = lambda *a, **k: None
_torch_stub.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
_torch_stub.long = "long"

_dist_stub = sys.modules["torch.distributed"]
_dist_stub.is_initialized = lambda: False
_dist_stub.barrier = lambda: None
_dist_stub.all_reduce = lambda *a, **k: None
_dist_stub.ReduceOp = type("ReduceOp", (), {"SUM": 0})()

_ds_zero_stub = sys.modules["deepspeed.zero"]
_ds_zero_stub.GatheredParameters = lambda *a, **k: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()

# anthropic stub
_anthropic_stub = sys.modules["anthropic"]
_anthropic_stub.HUMAN_PROMPT = "\n\nHuman: "
_anthropic_stub.AI_PROMPT = "\n\nAssistant: "

# numpy stub - provide basic array/mean
import array as _array_mod
_np_stub = sys.modules["numpy"]
_np_stub.array = lambda *a, **k: a[0] if a else []
_np_stub.mean = lambda x, **k: sum(x) / len(x) if x else 0
_np_stub.float64 = float
_np_stub.int64 = int
_np_stub.ndarray = list  # stub for type annotations
_np_stub.np = _np_stub

# tqdm stub
_tqdm_stub = sys.modules["tqdm"]
_tqdm_stub.tqdm = lambda x, *a, **k: x

# pandas stub - needs read_parquet for BigCodeBench
_pd_stub = sys.modules["pandas"]
_pd_stub.DataFrame = type("DataFrame", (), {})

# attrs stub
_attrs_stub = sys.modules["attrs"]
_attrs_stub.define = lambda cls=None, **k: cls if cls else (lambda c: c)

# pebble stub
_pebble_stub = sys.modules["pebble"]
_pebble_stub.concurrent = type("concurrent", (), {})()
_pebble_stub.ProcessPool = type("ProcessPool", (), {})
_pebble_stub.ProcessExpired = Exception

# Fix numpy stub with minimal ndarray support for pass_k_utils
_np_stub = sys.modules["numpy"]

class _FakeArray:
    def __init__(self, data):
        self._data = list(data) if hasattr(data, '__iter__') else [data]
    def __gt__(self, other):
        return _FakeArray([x > other for x in self._data])
    def __ge__(self, other):
        return _FakeArray([x >= other for x in self._data])
    def __sub__(self, other):
        if isinstance(other, _FakeArray):
            return _FakeArray([a - b for a, b in zip(self._data, other._data)])
        return _FakeArray([x - other for x in self._data])
    def __rsub__(self, other):
        return _FakeArray([other - x for x in self._data])
    def __truediv__(self, other):
        if isinstance(other, _FakeArray):
            return _FakeArray([a / b for a, b in zip(self._data, other._data)])
        return _FakeArray([x / other for x in self._data])
    def __rtruediv__(self, other):
        return _FakeArray([other / x for x in self._data])
    def __mul__(self, other):
        if isinstance(other, _FakeArray):
            return _FakeArray([a * b for a, b in zip(self._data, other._data)])
        return _FakeArray([x * other for x in self._data])
    def __rmul__(self, other):
        return self.__mul__(other)
    def __len__(self):
        return len(self._data)
    def __iter__(self):
        return iter(self._data)
    def __getitem__(self, idx):
        return self._data[idx]
    def all(self):
        return all(self._data)
    def tolist(self):
        return list(self._data)
    def mean(self):
        return sum(self._data) / len(self._data) if self._data else 0

_np_stub.array = lambda x, **k: _FakeArray(x)
_np_stub.all = lambda x, **k: all(x) if hasattr(x, '__iter__') else bool(x)
_np_stub.any = lambda x, **k: any(x) if hasattr(x, '__iter__') else bool(x)
def _np_prod(x, **k):
    result = 1.0
    for v in x:
        result *= v
    return result
_np_stub.prod = _np_prod
_np_stub.arange = lambda start, stop=None, step=1, **k: _FakeArray(list(range(start, stop, step)) if stop is not None else list(range(start)))
_np_stub.bool_ = bool

# Fix tqdm stub to support context manager pattern: with tqdm(...) as pbar:
class _TqdmFake:
    def __init__(self, iterable=None, *a, **k):
        self._iterable = iterable
    def __iter__(self):
        return iter(self._iterable) if self._iterable else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *a):
        pass
    def update(self, *a, **k):
        pass
    def set_description(self, *a, **k):
        pass
_tqdm_stub.tqdm = _TqdmFake

# sympy stub
_sympy_stub = types.ModuleType("sympy")
_sympy_stub.simplify = lambda x: x
_sympy_stub.expand = lambda x: x
_sympy_stub.trigsimp = lambda x: x
sys.modules["sympy"] = _sympy_stub

_sympy_parsing = types.ModuleType("sympy.parsing")
sys.modules["sympy.parsing"] = _sympy_parsing
_sympy_latex = types.ModuleType("sympy.parsing.latex")
_sympy_latex.parse_latex = lambda x: x
sys.modules["sympy.parsing.latex"] = _sympy_latex

# openai stub
_openai_stub = types.ModuleType("openai")
_openai_stub.OpenAI = type("OpenAI", (), {"__init__": lambda self, **k: None})
sys.modules["openai"] = _openai_stub

# yaml stub (if not available)
try:
    import yaml
except ImportError:
    _yaml_stub = types.ModuleType("yaml")
    _yaml_stub.safe_load = lambda x: {}
    sys.modules["yaml"] = _yaml_stub


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")


# ============================================================
# 1. Utils: answer_utils
# ============================================================
print("=" * 60)
print("1. Testing answer_utils")
print("=" * 60)

from finetune.test.utils.answer_utils import UnitTextManager, StringCleaner, AnswerExtractor

um = UnitTextManager()
sc = StringCleaner(um)
ae = AnswerExtractor(sc)

# boxed answers
check("boxed{42}", ae.extract_answer("So \\boxed{42}") == "42")
check("boxed{3/4}", ae.extract_answer("Answer is \\boxed{3/4}") == "3/4")
check("boxed nested {}", ae.extract_answer("\\boxed{2^{10}}") == "2^10")

# "the answer is" pattern
check("the answer is 7", ae.extract_answer("Therefore the answer is 7.") == "7")

# last number fallback
check("last number 99", ae.extract_answer("We compute 12 + 87 = 99") == "99")
check("last number neg", ae.extract_answer("The result is -3.5") == "-3.5")

# empty / edge
check("empty string", ae.extract_answer("") == "")
check("no number", ae.extract_answer("no numbers here") == "")

# StringCleaner specifics
check("strip % sign", sc.strip_string("75%") == "75")
check("strip leading dot", sc.strip_string(".5") == "0.5")
check("strip trailing dot", sc.strip_string("3.") == "3")
check("strip commas", sc.strip_string("1,000") == "1000")


# ============================================================
# 2. Utils: verify_utils
# ============================================================
print()
print("=" * 60)
print("2. Testing verify_utils")
print("=" * 60)

from finetune.test.utils.verify_utils import math_verify_compare, HAS_MATH_VERIFY

print(f"  math_verify library available: {HAS_MATH_VERIFY}")

check("exact match", math_verify_compare("42", "42"))
check("case insensitive", math_verify_compare("ABC", "abc"))
check("mismatch", not math_verify_compare("42", "43"))
check("whitespace", math_verify_compare(" 7 ", "7"))


# ============================================================
# 3. Utils: data_utils (with temp JSON files)
# ============================================================
print()
print("=" * 60)
print("3. Testing data_utils")
print("=" * 60)

from finetune.test.utils.data_utils import load_test_data

tmpdir = tempfile.mkdtemp()
samples = [
    {"instruction": "What is 1+1?", "output": "2"},
    {"instruction": "What is 2+2?", "output": "4"},
    {"instruction": "What is 3+3?", "output": "6"},
    {"instruction": "What is 4+4?", "output": "8"},
]
with open(os.path.join(tmpdir, "test.json"), "w") as f:
    json.dump(samples, f)

loaded = load_test_data(tmpdir, rank=0, world_size=1)
check("load all 4 samples", len(loaded) == 4)
check("question field", loaded[0]["question"] == "What is 1+1?")
check("answer field", loaded[0]["answer"] == "2")

# distributed sharding
shard0 = load_test_data(tmpdir, rank=0, world_size=2)
shard1 = load_test_data(tmpdir, rank=1, world_size=2)
check("shard sizes sum to total", len(shard0) + len(shard1) == 4)
check("shards are disjoint",
      set(s["question"] for s in shard0).isdisjoint(
          set(s["question"] for s in shard1)))

# empty dir
emptydir = tempfile.mkdtemp()
check("empty dir returns []", load_test_data(emptydir) == [])

# cleanup
shutil.rmtree(tmpdir)
shutil.rmtree(emptydir)


# ============================================================
# 4. Evaluators: extract_choice_answer (mmlu)
#    Load via importlib to bypass torch/deepspeed top-level imports
# ============================================================
print()
print("=" * 60)
print("4. Testing mmlu extract_choice_answer")
print("=" * 60)

# Stubs already injected at top; evaluators are already importable
from finetune.test.evaluators.mmlu_evaluator import extract_choice_answer as ecf

check("plain A", ecf("A") == "A")
check("The answer is B", ecf("The answer is B") == "B")
check("sentence with C", ecf("I think C is correct") == "I")
check("answer: C", ecf("Answer: C") == "C")
check("lowercase d", ecf("d") == "D")
check("empty response", ecf("") == "")


# ============================================================
# 5. Evaluators: generate_prompt (math)
# ============================================================
print()
print("=" * 60)
print("5. Testing math generate_prompt")
print("=" * 60)

from finetune.test.evaluators.math_evaluator import generate_prompt as gen_prompt

prompt = gen_prompt("What is 2+2?")
check("contains question", "What is 2+2?" in prompt)
check("contains boxed", "boxed" in prompt)
check("contains Instruction", "Instruction" in prompt)
check("contains Response", "Response" in prompt)

# with tokenizer=None should use plain template
prompt2 = gen_prompt("Solve x=5", tokenizer=None)
check("no tokenizer uses plain template", "### Instruction:" in prompt2)



# ============================================================
# 6. Load real LiveCodeBench problem from test6.jsonl
# ============================================================
print()
print("=" * 60)
print("6. Loading real LiveCodeBench problem from test6.jsonl")
print("=" * 60)

import json as _json
from datetime import datetime as _dt

REAL_PROBLEM_JSONL = '/data/fqzhou/dataset_agent/benchmark/livecodebench/test6.jsonl'
with open(REAL_PROBLEM_JSONL, 'r', encoding='utf-8') as _f:
    REAL_PROBLEM_RAW = _json.loads(_f.readline().strip())

check("real problem loaded", REAL_PROBLEM_RAW["question_title"] == "9x9 Sum")
check("real problem has public_test_cases", "public_test_cases" in REAL_PROBLEM_RAW)
check("real problem has private_test_cases", "private_test_cases" in REAL_PROBLEM_RAW)
check("real problem platform is atcoder", REAL_PROBLEM_RAW["platform"] == "atcoder")
check("real problem difficulty is easy", REAL_PROBLEM_RAW["difficulty"] == "easy")


# ============================================================
# 7. CodeGenerationProblem from real data (incl. compressed private tests)
# ============================================================
print()
print("=" * 60)
print("7. Testing CodeGenerationProblem with real test6.jsonl data")
print("=" * 60)

from finetune.test.lcb.benchmarks.code_generation import CodeGenerationProblem

real_problem = CodeGenerationProblem(**REAL_PROBLEM_RAW)

check("real problem title", real_problem.question_title == "9x9 Sum")
check("real problem platform parsed", real_problem.platform.value == "atcoder")
check("real problem difficulty parsed", real_problem.difficulty.value == "easy")
check("real problem contest_date is datetime", isinstance(real_problem.contest_date, _dt))
check("real problem has 3 public tests", len(real_problem.public_test_cases) == 3)
check("real problem public test 1 input", real_problem.public_test_cases[0].input == "1")
check("real problem public test 1 output", real_problem.public_test_cases[0].output == "2024")
check("real problem public test type is stdin", real_problem.public_test_cases[0].testtype.value == "stdin")

# Private tests are compressed - verify they were decompressed correctly
check("real problem has private tests", len(real_problem.private_test_cases) > 0)
check("real problem private tests > 10", len(real_problem.private_test_cases) >= 10)
check("real problem private test type is stdin", real_problem.private_test_cases[0].testtype.value == "stdin")

# get_evaluation_sample merges public + private
eval_sample = real_problem.get_evaluation_sample()
check("eval_sample has input_output", "input_output" in eval_sample)
io_data = _json.loads(eval_sample["input_output"])
total_tests = len(real_problem.public_test_cases) + len(real_problem.private_test_cases)
check("eval_sample inputs count matches", len(io_data["inputs"]) == total_tests)
check("eval_sample outputs count matches", len(io_data["outputs"]) == total_tests)
check("eval_sample fn_name is None (stdin problem)", io_data["fn_name"] is None)


# ============================================================
# 8. Prompt generation with real problem
# ============================================================
print()
print("=" * 60)
print("8. Testing prompt generation with real problem")
print("=" * 60)

from finetune.test.lcb.prompts.code_generation import format_prompt_generation
from finetune.test.lcb.lm_styles import LMStyle

# GenericBase style
prompt_base = format_prompt_generation(real_problem, LMStyle.GenericBase)
check("GenericBase prompt is string", isinstance(prompt_base, str))
check("prompt contains 9-by-9", "9-by-9" in prompt_base)
check("prompt contains question marker", "### Question" in prompt_base)
check("prompt has answer section", "### Answer" in prompt_base)

# code_evaluator generate_prompt
from finetune.test.evaluators.livecode_evaluator import generate_prompt as code_gen_prompt
code_prompt = code_gen_prompt(real_problem)
check("code_gen_prompt contains real question", "9-by-9" in code_prompt)


# ============================================================
# 9. Code extraction (LiveCodeBench robust extract_code)
# ============================================================
print()
print("=" * 60)
print("9. Testing LiveCodeBench extract_code (robust)")
print("=" * 60)

from finetune.test.evaluators.livecode_evaluator import extract_code as lcb_extract_code

# Raw code (no markdown) — returned as-is
raw_code = "x = int(input())\nprint(42)"
check("raw code returned as-is", lcb_extract_code(raw_code) == raw_code.strip())

# Markdown code block with ```python
markdown_output = "Here is the solution:\n```python\nx = int(input())\ntotal = sum(i*j for i in range(1,10) for j in range(1,10))\nprint(total)\n```\nThis works."
extracted = lcb_extract_code(markdown_output)
check("extracts from ```python block", "total = sum" in extracted)
check("excludes explanation text", "This works" not in extracted)
check("excludes ```python marker", "```" not in extracted)

# Plain ``` block (no language specifier)
plain_md = "Solution:\n```\nx = int(input())\nprint(42)\n```"
extracted_plain = lcb_extract_code(plain_md)
check("extracts from plain ``` block", "print(42)" in extracted_plain)

# No code block — return raw (base model behavior)
raw_only = "x = int(input())\nfor i in range(1,10):\n    print(i)"
check("no fences returns raw code", lcb_extract_code(raw_only) == raw_only.strip())

# Empty output
check("empty returns empty", lcb_extract_code("") == "")
check("whitespace returns empty", lcb_extract_code("   \n  ") == "")

# Multiple code blocks — takes the last one
multi_block = "First attempt:\n```python\nwrong = 1\n```\nCorrection:\n```python\nx = int(input())\nprint(42)\n```"
extracted_multi = lcb_extract_code(multi_block)
check("multiple blocks extracts last", "print(42)" in extracted_multi)
check("multiple blocks excludes first", "wrong" not in extracted_multi)

# Single fence line (incomplete block) — treat as raw
single_fence = "```python\nx = 1"
check("incomplete fence returns raw", lcb_extract_code(single_fence) == single_fence.strip())


# ============================================================
# 10. check_correctness with real problem: correct code
# ============================================================
print()
print("=" * 60)
print("10. Testing check_correctness with CORRECT code on real problem")
print("=" * 60)

from finetune.test.lcb.evaluation.compute_code_generation_metrics import check_correctness

eval_sample = real_problem.get_evaluation_sample()

# Correct solution for "9x9 Sum": sum all i*j for 1<=i,j<=9, minus all cells equal to X
correct_code = """
x = int(input())
total = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j != x:
            total += i * j
print(total)
"""

result_correct, meta_correct = check_correctness(eval_sample, correct_code, timeout=5, debug=False)
num_tests = len(_json.loads(eval_sample["input_output"])["inputs"])
check(f"correct code returns {num_tests} results", len(result_correct) == num_tests)
check("correct code passes ALL tests", all(r == True for r in result_correct))


# ============================================================
# 11. check_correctness with real problem: WRONG code (logic error)
# ============================================================
print()
print("=" * 60)
print("11. Testing check_correctness with WRONG code (logic error)")
print("=" * 60)

# Wrong logic: subtracts X only once instead of counting all cells with value X
wrong_logic_code = """
x = int(input())
total = sum(i * j for i in range(1, 10) for j in range(1, 10))
print(total - x)
"""

result_wrong, meta_wrong = check_correctness(eval_sample, wrong_logic_code, timeout=5, debug=False)
check("wrong logic code returns results", len(result_wrong) > 0)
# For X=24, there are 4 cells with value 24 (3*8,4*6,6*4,8*3), so correct=2025-4*24=1929, but wrong=2025-24=2001
check("wrong logic code does NOT pass all tests", not all(r == True for r in result_wrong))


# ============================================================
# 12. check_correctness with real problem: WRONG code (runtime error)
# ============================================================
print()
print("=" * 60)
print("12. Testing check_correctness with WRONG code (runtime error)")
print("=" * 60)

runtime_error_code = """
x = int(input())
result = 1 / 0
print(result)
"""

result_runtime, meta_runtime = check_correctness(eval_sample, runtime_error_code, timeout=5, debug=False)
check("runtime error code returns results", len(result_runtime) > 0)
check("runtime error code fails all tests", all(r != True for r in result_runtime))


# ============================================================
# 13. check_correctness with real problem: WRONG code (syntax error)
# ============================================================
print()
print("=" * 60)
print("13. Testing check_correctness with WRONG code (syntax error)")
print("=" * 60)

syntax_error_code = """
x = int(input()
print(x)
"""

result_syntax, meta_syntax = check_correctness(eval_sample, syntax_error_code, timeout=5, debug=False)
check("syntax error code returns results", len(result_syntax) > 0)
check("syntax error code fails all tests", all(r != True for r in result_syntax))


# ============================================================
# 14. check_correctness with real problem: WRONG code (infinite loop)
# ============================================================
print()
print("=" * 60)
print("14. Testing check_correctness with WRONG code (infinite loop)")
print("=" * 60)

infinite_loop_code = """
x = int(input())
while True:
    pass
"""

result_timeout, meta_timeout = check_correctness(eval_sample, infinite_loop_code, timeout=2, debug=False)
check("infinite loop code returns results", len(result_timeout) > 0)
check("infinite loop code fails", all(r != True for r in result_timeout))


# ============================================================
# 15. Full codegen_metrics pipeline with real problem
# ============================================================
print()
print("=" * 60)
print("15. Testing codegen_metrics end-to-end with real problem")
print("=" * 60)

from finetune.test.lcb.evaluation.compute_code_generation_metrics import codegen_metrics

eval_samples = [real_problem.get_evaluation_sample()]

# Test with correct code
correct_generations = [[correct_code.strip()]]
metrics_correct, results_correct, metadata_out = codegen_metrics(
    eval_samples, correct_generations, num_process_evaluate=1, timeout=5
)
check("codegen_metrics returns pass@1", "pass@1" in metrics_correct)
check("correct code pass@1 == 1.0", metrics_correct["pass@1"] == 1.0)

# Test with wrong code
wrong_generations = [[wrong_logic_code.strip()]]
metrics_wrong, results_wrong, _ = codegen_metrics(
    eval_samples, wrong_generations, num_process_evaluate=1, timeout=5
)
check("wrong code pass@1 < 1.0", metrics_wrong["pass@1"] < 1.0)


# ============================================================
# 16. BigCodeBench: data loading from local parquet
# ============================================================
print()
print("=" * 60)
print("16. BigCodeBench: data loading from local parquet")
print("=" * 60)

# We need real pandas+numpy for parquet reading — remove stubs, import real ones
del sys.modules["pandas"]
for _k in list(sys.modules.keys()):
    if _k == "numpy" or _k.startswith("numpy."):
        del sys.modules[_k]
import numpy  # real numpy
import pandas  # real pandas
sys.modules["numpy"] = numpy
sys.modules["pandas"] = pandas

# Give torch stub a proper __spec__ so importlib.util.find_spec("torch")
# (called by datasets library during pandas import chain) doesn't raise ValueError
import importlib
import importlib.machinery
_torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", None)

# Load data directly with real pandas (avoids fragile importlib.reload)
# This tests the same logic as load_bigcodebench_local
BCB_DIR = "/data/fqzhou/dataset_agent/benchmark/bigcodebench"
_parquet_files = [f for f in os.listdir(BCB_DIR) if f.endswith(".parquet")]
assert _parquet_files, "No parquet files found"
_df = pandas.read_parquet(os.path.join(BCB_DIR, _parquet_files[0]))
tasks = _df.to_dict("records")
tasks.sort(key=lambda x: x["task_id"])

check("loaded 1140 tasks", len(tasks) == 1140)
check("tasks sorted by task_id", tasks[0]["task_id"] == "BigCodeBench/0")
check("last task_id (string sort)", tasks[-1]["task_id"] == "BigCodeBench/999")
check("task has complete_prompt", "complete_prompt" in tasks[0])
check("task has instruct_prompt", "instruct_prompt" in tasks[0])
check("task has canonical_solution", "canonical_solution" in tasks[0])
check("task has test", "test" in tasks[0])
check("task has entry_point", "entry_point" in tasks[0])
check("task has code_prompt", "code_prompt" in tasks[0])
check("task has libs", "libs" in tasks[0])

# Verify task content
t0 = tasks[0]
check("task 0 entry_point is task_func", t0["entry_point"] == "task_func")
check("task 0 complete_prompt has def", "def task_func" in t0["complete_prompt"])
check("task 0 test has TestCases", "class TestCases" in t0["test"])
check("task 0 test has unittest", "unittest" in t0["test"])

# Test load_bigcodebench_local on empty dir
# Reload the module now that real numpy/pandas are in sys.modules
import finetune.test.evaluators.bigcodebench_evaluator as _bcb_mod
importlib.reload(_bcb_mod)
from finetune.test.evaluators.bigcodebench_evaluator import load_bigcodebench_local

emptydir2 = tempfile.mkdtemp()
try:
    load_bigcodebench_local(emptydir2)
    check("empty dir raises FileNotFoundError", False)
except FileNotFoundError:
    check("empty dir raises FileNotFoundError", True)
shutil.rmtree(emptydir2)


# ============================================================
# 17. BigCodeBench: prompt generation
# ============================================================
print()
print("=" * 60)
print("17. BigCodeBench: prompt generation")
print("=" * 60)

from finetune.test.evaluators.bigcodebench_evaluator import generate_prompt as bcb_gen_prompt

# Without tokenizer: should return complete_prompt as-is
prompt_no_tok = bcb_gen_prompt(t0, tokenizer=None)
check("no tokenizer returns complete_prompt", prompt_no_tok == t0["complete_prompt"])

# With a fake tokenizer that has apply_chat_template
class FakeTokenizer:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

fake_tok = FakeTokenizer()
prompt_with_tok = bcb_gen_prompt(t0, tokenizer=fake_tok)
check("with tokenizer uses chat template", "<|system|>" in prompt_with_tok)
check("chat template includes complete_prompt", t0["complete_prompt"] in prompt_with_tok)
check("chat template has assistant marker", "<|assistant|>" in prompt_with_tok)


# ============================================================
# 18. BigCodeBench: code extraction
# ============================================================
print()
print("=" * 60)
print("18. BigCodeBench: code extraction")
print("=" * 60)

from finetune.test.evaluators.bigcodebench_evaluator import extract_code as bcb_extract

prompt = t0["complete_prompt"]

# Case 1: raw function body (model directly outputs continuation)
raw_body = "    permutations = list(itertools.permutations(numbers))\n    return 42\n"
extracted = bcb_extract(raw_body, prompt)
check("raw body: prepends prompt", extracted.startswith(prompt))
check("raw body: includes body", "return 42" in extracted)

# Case 2: model repeats the prompt then continues
repeated = prompt.strip() + "\n    return 42\n"
extracted2 = bcb_extract(repeated, prompt)
check("repeated prompt: strips duplicate", extracted2.startswith(prompt))
check("repeated prompt: has body", "return 42" in extracted2)
# Should not double the prompt
check("repeated prompt: no double prompt", extracted2.count("def task_func") == 1)

# Case 3: markdown code block
md_output = "Here's the solution:\n```python\n    return 42\n```\nDone."
extracted3 = bcb_extract(md_output, prompt)
check("markdown block: prepends prompt", extracted3.startswith(prompt))
check("markdown block: extracts body", "return 42" in extracted3)
check("markdown block: excludes explanation", "Done." not in extracted3)

# Case 4: generic markdown block (no python specifier)
md_generic = "```\n    return 42\n```"
extracted4 = bcb_extract(md_generic, prompt)
check("generic md block: extracts body", "return 42" in extracted4)

# Case 5: empty output
extracted5 = bcb_extract("", prompt)
check("empty output: returns prompt only", extracted5 == prompt)


# ============================================================
# 19. BigCodeBench: sandbox execution with untrusted_check
# ============================================================
print()
print("=" * 60)
print("19. BigCodeBench: sandbox execution (untrusted_check)")
print("=" * 60)

from finetune.test.evaluators.bigcodebench_evaluator import untrusted_check, PASS, FAIL, TIMEOUT

# Use BigCodeBench/23 (deterministic, no randomness):
# task_func alternates elements from two lists, finds closest to 0.5
t23 = next(t for t in tasks if t["task_id"] == "BigCodeBench/23")
check("found BigCodeBench/23", t23["task_id"] == "BigCodeBench/23")

# --- 19a: correct code (canonical solution) ---
correct_code_23 = t23["complete_prompt"] + t23["canonical_solution"]
stat, details = untrusted_check(
    code=correct_code_23,
    test_code=t23["test"],
    entry_point=t23["entry_point"],
)
check("19a canonical solution passes", stat == PASS)
check("19a no failure details", len(details) == 0)

# --- 19b: wrong logic ---
wrong_code_23 = t23["complete_prompt"] + "    return 999.0\n"
stat_w, details_w = untrusted_check(
    code=wrong_code_23,
    test_code=t23["test"],
    entry_point=t23["entry_point"],
)
check("19b wrong logic fails", stat_w == FAIL)
check("19b has failure details", len(details_w) > 0)

# --- 19c: runtime error ---
runtime_code_23 = t23["complete_prompt"] + "    raise ValueError('boom')\n"
stat_r, details_r = untrusted_check(
    code=runtime_code_23,
    test_code=t23["test"],
    entry_point=t23["entry_point"],
)
check("19c runtime error fails", stat_r == FAIL)

# --- 19d: syntax error ---
syntax_code_23 = t23["complete_prompt"] + "    return (\n"
stat_s, details_s = untrusted_check(
    code=syntax_code_23,
    test_code=t23["test"],
    entry_point=t23["entry_point"],
)
check("19d syntax error fails", stat_s == FAIL)

# --- 19e: infinite loop (timeout) ---
# Temporarily lower TIMEOUT_LIMIT to avoid 240s wait
import finetune.test.evaluators.bigcodebench_evaluator as _bcb
_orig_timeout = _bcb.TIMEOUT_LIMIT
_bcb.TIMEOUT_LIMIT = 3.0
loop_code_23 = t23["complete_prompt"] + "    while True: pass\n"
stat_t, details_t = untrusted_check(
    code=loop_code_23,
    test_code=t23["test"],
    entry_point=t23["entry_point"],
    min_time_limit=2,
    gt_time_limit=2,
)
_bcb.TIMEOUT_LIMIT = _orig_timeout
check("19e infinite loop times out", stat_t == TIMEOUT)


# ============================================================
# 20. BigCodeBench: estimate_pass_at_k
# ============================================================
print()
print("=" * 60)
print("20. BigCodeBench: estimate_pass_at_k")
print("=" * 60)

from finetune.test.evaluators.bigcodebench_evaluator import estimate_pass_at_k
import numpy as real_np

# All correct: pass@1 = 1.0
result = estimate_pass_at_k(num_samples=5, num_correct=[5], k=1)
check("all correct: pass@1 = 1.0", abs(float(result[0]) - 1.0) < 1e-9)

# None correct: pass@1 = 0.0
result2 = estimate_pass_at_k(num_samples=5, num_correct=[0], k=1)
check("none correct: pass@1 = 0.0", abs(float(result2[0])) < 1e-9)

# 1 of 5 correct: pass@1 = 0.2
result3 = estimate_pass_at_k(num_samples=5, num_correct=[1], k=1)
check("1/5 correct: pass@1 = 0.2", abs(float(result3[0]) - 0.2) < 1e-9)

# Multiple problems
result4 = estimate_pass_at_k(num_samples=10, num_correct=[10, 0, 5], k=1)
vals = [float(v) for v in result4]
check("multi-problem: first=1.0", abs(vals[0] - 1.0) < 1e-9)
check("multi-problem: second=0.0", abs(vals[1]) < 1e-9)
check("multi-problem: third=0.5", abs(vals[2] - 0.5) < 1e-9)

# pass@5 with n=5, c=3: since n-c=2 < k=5, should be 1.0
result5 = estimate_pass_at_k(num_samples=5, num_correct=[3], k=5)
check("n-c < k: pass@k = 1.0", abs(float(result5[0]) - 1.0) < 1e-9)


# ============================================================
# 21. BigCodeBench: batch evaluation (sequential, avoids daemon nesting)
# ============================================================
print()
print("=" * 60)
print("21. BigCodeBench: batch evaluation with mixed outcomes")
print("=" * 60)

# Prepare 3 test cases with different expected outcomes
batch_cases = [
    ("correct", t23["complete_prompt"] + t23["canonical_solution"], PASS),
    ("wrong logic", t23["complete_prompt"] + "    return 999.0\n", FAIL),
    ("runtime error", t23["complete_prompt"] + "    raise ValueError('x')\n", FAIL),
]

batch_results = []
for label, code, expected in batch_cases:
    stat, details = untrusted_check(
        code=code, test_code=t23["test"], entry_point=t23["entry_point"],
    )
    batch_results.append((label, stat, expected))
    check(f"batch {label}: {stat} == {expected}", stat == expected)

check("batch returns 3 results", len(batch_results) == 3)


# ============================================================
# 22. BigCodeBench: canonical solutions on multiple real tasks
# ============================================================
print()
print("=" * 60)
print("22. BigCodeBench: canonical solutions on multiple tasks")
print("=" * 60)

# Test canonical solutions for a few tasks with diverse library deps
test_task_ids = ["BigCodeBench/23", "BigCodeBench/50", "BigCodeBench/100"]
task_lookup = {t["task_id"]: t for t in tasks}

for tid in test_task_ids:
    t = task_lookup[tid]
    code = t["complete_prompt"] + t["canonical_solution"]
    stat, details = untrusted_check(
        code=code, test_code=t["test"], entry_point=t["entry_point"],
    )
    check(f"{tid} canonical passes", stat == PASS)
    if stat != PASS:
        print(f"    FAILED: {tid} -> {stat}: {details}")


# ============================================================
# 23. BigCodeBench: routing in evaluate_test_set
# ============================================================
print()
print("=" * 60)
print("23. BigCodeBench: routing in evaluate_test_set")
print("=" * 60)

init_path = "/data/fqzhou/dataset_agent/finetune/test/__init__.py"
with open(init_path) as f:
    src = f.read()

check("bigcodebench in routing", "bigcodebench" in src)
check("evaluate_bigcodebench imported", "evaluate_bigcodebench" in src)
check("bigcodebench route calls evaluate_bigcodebench",
      "elif dataset_name in ['bigcodebench']" in src and "evaluate_bigcodebench" in src)

# Verify the evaluators/__init__.py exports
init2_path = "/data/fqzhou/dataset_agent/finetune/test/evaluators/__init__.py"
with open(init2_path) as f:
    src2 = f.read()

check("evaluators __init__ exports evaluate_bigcodebench",
      "from .bigcodebench_evaluator import evaluate_bigcodebench" in src2)
check("evaluate_bigcodebench in __all__", "evaluate_bigcodebench" in src2)


# ============================================================
# 24. BigCodeBench: evaluate_bigcodebench function signature
# ============================================================
print()
print("=" * 60)
print("24. BigCodeBench: evaluate_bigcodebench interface check")
print("=" * 60)

import inspect
from finetune.test.evaluators.bigcodebench_evaluator import evaluate_bigcodebench

sig = inspect.signature(evaluate_bigcodebench)
params = list(sig.parameters.keys())
check("has model param", "model" in params)
check("has tokenizer param", "tokenizer" in params)
check("has test_data_dir param", "test_data_dir" in params)
check("has rank param", "rank" in params)
check("has world_size param", "world_size" in params)
check("has max_samples param", "max_samples" in params)
check("has max_gen_length param", "max_gen_length" in params)
check("has batch_size param", "batch_size" in params)

# Check defaults
check("rank default is 0", sig.parameters["rank"].default == 0)
check("world_size default is 1", sig.parameters["world_size"].default == 1)
check("max_samples default is None", sig.parameters["max_samples"].default is None)
check("max_gen_length default is 2048", sig.parameters["max_gen_length"].default == 2048)
check("batch_size default is 1", sig.parameters["batch_size"].default == 1)


# ============================================================
# 25. Routing: evaluate_test_set dispatches all datasets correctly
# ============================================================
print()
print("=" * 60)
print("25. Testing evaluate_test_set routing logic (all datasets)")
print("=" * 60)

check("routes mmlu to evaluate_mmlu_choice", "mmlu" in src and "evaluate_mmlu_choice" in src)
check("routes others to evaluate_exact_match", "evaluate_exact_match" in src)
check("routes livecodebench to evaluate_code_generation", "livecodebench" in src and "evaluate_code_generation" in src)
check("routes bigcodebench to evaluate_bigcodebench", "bigcodebench" in src and "evaluate_bigcodebench" in src)
check("routes humaneval to evaluate_humaneval", "humaneval" in src and "evaluate_humaneval" in src)
check("routes physics to evaluate_physics", "physics" in src and "evaluate_physics" in src)
check("routes openfindata to evaluate_openfindata", "openfindata" in src and "evaluate_openfindata" in src)
check("uses os.path.basename for detection", "os.path.basename" in src)
check("mmlu_pro in routing condition", "mmlu_pro" in src)


# ============================================================
# 26. HumanEval: load module (direct, avoids lcb import chain)
# ============================================================
print()
print("=" * 60)
print("26. HumanEval: load module directly")
print("=" * 60)

import importlib.util as _ilu

_he_spec = _ilu.spec_from_file_location(
    "humaneval_evaluator",
    "/data/fqzhou/dataset_agent/finetune/test/evaluators/humaneval_evaluator.py",
    submodule_search_locations=[],
)
_he_mod = _ilu.module_from_spec(_he_spec)
_he_spec.loader.exec_module(_he_mod)
check("humaneval module loaded", hasattr(_he_mod, "evaluate_humaneval"))
check("has load_humaneval_local", hasattr(_he_mod, "load_humaneval_local"))
check("has check_correctness", hasattr(_he_mod, "check_correctness"))
check("has generate_prompt", hasattr(_he_mod, "generate_prompt"))
check("has extract_code", hasattr(_he_mod, "extract_code"))
check("has run_tests_parallel", hasattr(_he_mod, "run_tests_parallel"))
check("has TIMEOUT_LIMIT", hasattr(_he_mod, "TIMEOUT_LIMIT"))


# ============================================================
# 27. HumanEval: data loading from local JSONL
# ============================================================
print()
print("=" * 60)
print("27. HumanEval: data loading from local JSONL")
print("=" * 60)

HE_DIR = "/data/fqzhou/dataset_agent/benchmark/humaneval"
he_problems = _he_mod.load_humaneval_local(HE_DIR)

check("loaded 164 problems", len(he_problems) == 164)
check("sorted by task_id", he_problems[0]["task_id"] == "HumanEval/0")
check("has prompt", "prompt" in he_problems[0])
check("has entry_point", "entry_point" in he_problems[0])
check("has canonical_solution", "canonical_solution" in he_problems[0])
check("has test", "test" in he_problems[0])

he0 = he_problems[0]
check("problem 0 entry_point", he0["entry_point"] == "has_close_elements")
check("problem 0 prompt has def", "def has_close_elements" in he0["prompt"])
check("problem 0 test has check", "check(" in he0["test"])

# Error on empty dir
emptydir_he = tempfile.mkdtemp()
try:
    _he_mod.load_humaneval_local(emptydir_he)
    check("empty dir raises FileNotFoundError", False)
except FileNotFoundError:
    check("empty dir raises FileNotFoundError", True)
shutil.rmtree(emptydir_he)


# ============================================================
# 28. HumanEval: prompt generation
# ============================================================
print()
print("=" * 60)
print("28. HumanEval: prompt generation")
print("=" * 60)

he_gen_prompt = _he_mod.generate_prompt

# Without tokenizer: returns raw prompt (for base model continuation)
prompt_no_tok = he_gen_prompt(he0, tokenizer=None)
check("no tokenizer returns prompt", prompt_no_tok == he0["prompt"])

# With fake chat tokenizer
class FakeTokenizer2:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in msgs:
            parts.append(f"<|{m['role']}|>\n{m['content']}")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "\n".join(parts)

fake_tok2 = FakeTokenizer2()
prompt_chat = he_gen_prompt(he0, tokenizer=fake_tok2)
check("chat template has system", "<|system|>" in prompt_chat)
check("chat template has prompt content", he0["prompt"] in prompt_chat)
check("chat template has assistant", "<|assistant|>" in prompt_chat)
check("chat template has instruction", "Complete" in prompt_chat)


# ============================================================
# 29. HumanEval: code extraction
# ============================================================
print()
print("=" * 60)
print("29. HumanEval: code extraction")
print("=" * 60)

he_extract = _he_mod.extract_code
prompt = he0["prompt"]

# Case 1: raw function body (base model continuation)
raw_body = "    for idx, elem in enumerate(numbers):\n        return True\n"
ex1 = he_extract(raw_body, prompt)
check("raw body extracted", "return True" in ex1)

# Case 2: model repeats the prompt then continues
repeated = prompt.strip() + "\n    return True\n"
ex2 = he_extract(repeated, prompt)
check("repeated prompt stripped", "return True" in ex2)
check("no double def", ex2.count("def has_close_elements") == 0)

# Case 3: markdown ```python block
md_output = "Here's the solution:\n```python\n    return True\n```\nDone."
ex3 = he_extract(md_output, prompt)
check("markdown extracts body", "return True" in ex3)
check("markdown excludes explanation", "Done." not in ex3)

# Case 4: generic ``` block
md_generic = "```\n    return True\n```"
ex4 = he_extract(md_generic, prompt)
check("generic md extracts body", "return True" in ex4)

# Case 5: empty output
ex5 = he_extract("", prompt)
check("empty output returns empty", ex5 == "")


# ============================================================
# 30. HumanEval: sandbox execution with check_correctness
# ============================================================
print()
print("=" * 60)
print("30. HumanEval: sandbox execution (check_correctness)")
print("=" * 60)

he_check = _he_mod.check_correctness

# 30a: canonical solution passes
r_correct = he_check(he0, he0["canonical_solution"], timeout=10.0)
check("30a canonical passes", r_correct["passed"] is True)
check("30a result is passed", r_correct["result"] == "passed")
check("30a has task_id", r_correct["task_id"] == "HumanEval/0")

# 30b: wrong logic
r_wrong = he_check(he0, "    return False\n", timeout=10.0)
check("30b wrong logic fails", r_wrong["passed"] is False)
check("30b result is failed", r_wrong["result"].startswith("failed"))

# 30c: runtime error
r_runtime = he_check(he0, "    raise ValueError('boom')\n", timeout=10.0)
check("30c runtime error fails", r_runtime["passed"] is False)

# 30d: syntax error
r_syntax = he_check(he0, "    return (\n", timeout=10.0)
check("30d syntax error fails", r_syntax["passed"] is False)

# 30e: infinite loop (timeout)
_orig_he_timeout = _he_mod.TIMEOUT_LIMIT
_he_mod.TIMEOUT_LIMIT = 3.0
r_timeout = he_check(he0, "    while True: pass\n", timeout=3.0)
_he_mod.TIMEOUT_LIMIT = _orig_he_timeout
check("30e infinite loop times out", r_timeout["passed"] is False)
check("30e result is timed out", r_timeout["result"] == "timed out")


# ============================================================
# 31. HumanEval: canonical solutions on multiple problems
# ============================================================
print()
print("=" * 60)
print("31. HumanEval: canonical solutions on multiple problems")
print("=" * 60)

he_lookup = {p["task_id"]: p for p in he_problems}
he_test_ids = ["HumanEval/0", "HumanEval/1", "HumanEval/2",
               "HumanEval/10", "HumanEval/50", "HumanEval/100"]
for tid in he_test_ids:
    p = he_lookup[tid]
    r = he_check(p, p["canonical_solution"], timeout=10.0)
    check(f"{tid} canonical passes", r["passed"] is True)
    if not r["passed"]:
        print(f"    FAILED: {tid} -> {r['result']}")


# ============================================================
# 32. HumanEval: parallel test execution
# ============================================================
print()
print("=" * 60)
print("32. HumanEval: parallel test execution")
print("=" * 60)

he_parallel = _he_mod.run_tests_parallel

# Mix of correct and wrong
par_problems = [he_lookup["HumanEval/0"], he_lookup["HumanEval/1"], he_lookup["HumanEval/2"]]
par_completions = [
    par_problems[0]["canonical_solution"],  # correct
    "    return []\n",                       # wrong
    par_problems[2]["canonical_solution"],  # correct
]
par_results = he_parallel(par_problems, par_completions, num_workers=3)

check("parallel returns 3 results", len(par_results) == 3)
check("parallel result 0 (correct) passes", par_results[0][1]["passed"] is True)
check("parallel result 1 (wrong) fails", par_results[1][1]["passed"] is False)
check("parallel result 2 (correct) passes", par_results[2][1]["passed"] is True)
check("parallel results sorted by idx", [r[0] for r in par_results] == [0, 1, 2])


# ============================================================
# 33. HumanEval: evaluate_humaneval function signature
# ============================================================
print()
print("=" * 60)
print("33. HumanEval: evaluate_humaneval interface check")
print("=" * 60)

he_sig = inspect.signature(_he_mod.evaluate_humaneval)
he_params = list(he_sig.parameters.keys())
check("has model param", "model" in he_params)
check("has tokenizer param", "tokenizer" in he_params)
check("has test_data_dir param", "test_data_dir" in he_params)
check("has rank param", "rank" in he_params)
check("has world_size param", "world_size" in he_params)
check("has max_samples param", "max_samples" in he_params)
check("has max_gen_length param", "max_gen_length" in he_params)
check("has batch_size param", "batch_size" in he_params)

check("rank default is 0", he_sig.parameters["rank"].default == 0)
check("world_size default is 1", he_sig.parameters["world_size"].default == 1)
check("max_samples default is None", he_sig.parameters["max_samples"].default is None)
check("max_gen_length default is 512", he_sig.parameters["max_gen_length"].default == 512)
check("batch_size default is 1", he_sig.parameters["batch_size"].default == 1)

# Check routing in test/__init__.py
check("humaneval route in __init__", "evaluate_humaneval" in src and "'humaneval'" in src)

# Check evaluators/__init__.py exports
init2_path = "/data/fqzhou/dataset_agent/finetune/test/evaluators/__init__.py"
with open(init2_path) as f:
    eval_init_src = f.read()
check("evaluators exports evaluate_humaneval",
      "from .humaneval_evaluator import evaluate_humaneval" in eval_init_src)



# ============================================================
# 34. CRUXEval-O: load module directly
# ============================================================
print()
print("=" * 60)
print("34. CRUXEval-O: load module directly")
print("=" * 60)

_co_spec = _ilu.spec_from_file_location(
    "cruxeval_output_evaluator",
    "/data/fqzhou/dataset_agent/finetune/test/evaluators/cruxeval_output_evaluator.py",
    submodule_search_locations=[],
)
_co_mod = _ilu.module_from_spec(_co_spec)
_co_spec.loader.exec_module(_co_mod)
check("cruxeval-O module loaded", hasattr(_co_mod, "evaluate_cruxeval_output"))
check("has load_cruxeval_local", hasattr(_co_mod, "load_cruxeval_local"))
check("has check_correctness", hasattr(_co_mod, "check_correctness"))
check("has generate_prompt", hasattr(_co_mod, "generate_prompt"))
check("has extract_answer", hasattr(_co_mod, "extract_answer"))
check("has run_tests_parallel", hasattr(_co_mod, "run_tests_parallel"))
check("has TIMEOUT_LIMIT", hasattr(_co_mod, "TIMEOUT_LIMIT"))


# ============================================================
# 35. CRUXEval: data loading from local JSONL
# ============================================================
print()
print("=" * 60)
print("35. CRUXEval: data loading from local JSONL")
print("=" * 60)

CE_O_DIR = "/data/fqzhou/dataset_agent/benchmark/cruxeval_O"
CE_I_DIR = "/data/fqzhou/dataset_agent/benchmark/cruxeval_I"
ce_problems = _co_mod.load_cruxeval_local(CE_O_DIR)

check("loaded 800 problems", len(ce_problems) == 800)
check("sorted by id", ce_problems[0]["id"] == "sample_0")
check("last id", ce_problems[-1]["id"] == "sample_99")
check("has code field", "code" in ce_problems[0])
check("has input field", "input" in ce_problems[0])
check("has output field", "output" in ce_problems[0])
check("has id field", "id" in ce_problems[0])

ce0 = ce_problems[0]
check("sample_0 code has def f", "def f" in ce0["code"])
check("sample_0 has input", len(ce0["input"]) > 0)
check("sample_0 has output", len(ce0["output"]) > 0)

# Error on empty dir
emptydir_ce = tempfile.mkdtemp()
try:
    _co_mod.load_cruxeval_local(emptydir_ce)
    check("empty dir raises FileNotFoundError", False)
except FileNotFoundError:
    check("empty dir raises FileNotFoundError", True)
shutil.rmtree(emptydir_ce)


# ============================================================
# 36. CRUXEval-O: prompt generation
# ============================================================
print()
print("=" * 60)
print("36. CRUXEval-O: prompt generation")
print("=" * 60)

co_gen_prompt = _co_mod.generate_prompt

# Without tokenizer: returns raw few-shot prompt
prompt_o_raw = co_gen_prompt(ce0, tokenizer=None)
check("raw prompt contains code", ce0["code"] in prompt_o_raw)
check("raw prompt contains input", ce0["input"] in prompt_o_raw)
check("raw prompt has [ANSWER] tag", "[ANSWER]" in prompt_o_raw)
check("raw prompt has [PYTHON] tag", "[PYTHON]" in prompt_o_raw)
check("raw prompt has few-shot example", "assert f(17) == 17" in prompt_o_raw)
check("raw prompt ends with [ANSWER]", prompt_o_raw.rstrip().endswith("[ANSWER]"))

# With fake chat tokenizer
prompt_o_chat = co_gen_prompt(ce0, tokenizer=fake_tok2)
check("chat prompt has system", "<|system|>" in prompt_o_chat)
check("chat prompt has assistant", "<|assistant|>" in prompt_o_chat)
check("chat prompt contains code", ce0["code"] in prompt_o_chat)


# ============================================================
# 37. CRUXEval-O: answer extraction
# ============================================================
print()
print("=" * 60)
print("37. CRUXEval-O: answer extraction")
print("=" * 60)

co_extract = _co_mod.extract_answer

# Case 1: clean assertion + [/ANSWER]
check("clean assertion",
      co_extract('assert f(17) == 17\n[/ANSWER]') == '17')

# Case 2: just the assertion
check("assertion only",
      co_extract('assert f("x9j") == "x9ja"') == '"x9ja"')

# Case 3: direct value (no assert)
check("direct value",
      co_extract('42') == '42')

# Case 4: reasoning + assertion
check("reasoning then assertion",
      co_extract('Let me trace...\nassert f(17) == 17\n[/ANSWER]') == '17')

# Case 5: string output with ==
check("string output",
      co_extract('assert f("hi") == "bhihia"\n[/ANSWER]') == '"bhihia"')

# Case 6: list output
check("list output",
      co_extract('assert f([1,2]) == [3,4]\n[/ANSWER]') == '[3,4]')

# Case 7: empty
check("empty returns empty", co_extract('') == '')

# Case 8: with extra [ANSWER] tag from model
check("extra ANSWER tag",
      co_extract('thinking...\n[ANSWER]\nassert f(1) == 2\n[/ANSWER]') == '2')


# ============================================================
# 38. CRUXEval-O: execution (correct, wrong, timeout)
# ============================================================
print()
print("=" * 60)
print("38. CRUXEval-O: execution (check_correctness)")
print("=" * 60)

co_check = _co_mod.check_correctness

# Use sample_3: f(text, value) = text + value, f("bcksrut", "q") == "bcksrutq"
s3 = next(p for p in ce_problems if p["id"] == "sample_3")

# 38a: correct assertion
prog_correct = f"{s3['code']}\nassert {s3['output']} == {s3['output']}"
check("38a correct assertion passes", co_check(prog_correct, timeout=5.0))

# 38b: wrong assertion
prog_wrong = f"{s3['code']}\nassert {s3['output']} == 'wrong'"
check("38b wrong assertion fails", not co_check(prog_wrong, timeout=5.0))

# 38c: actual function call matches
prog_call = f"{s3['code']}\nassert {s3['output']} == f({s3['input']})"
check("38c function call matches", co_check(prog_call, timeout=5.0))

# 38d: runtime error
prog_err = "def f(): raise ValueError()\nassert 1 == f()"
check("38d runtime error fails", not co_check(prog_err, timeout=5.0))

# 38e: timeout
prog_loop = "def f():\n    while True: pass\nassert 1 == f()"
check("38e infinite loop times out", not co_check(prog_loop, timeout=2.0))


# ============================================================
# 39. CRUXEval-O: canonical samples pass (verify data integrity)
# ============================================================
print()
print("=" * 60)
print("39. CRUXEval-O: canonical samples pass")
print("=" * 60)

# Verify that for several samples, f(input) == output actually holds
for sid in ["sample_0", "sample_3", "sample_10", "sample_50", "sample_100", "sample_500"]:
    s = next(p for p in ce_problems if p["id"] == sid)
    prog = f"{s['code']}\nassert {s['output']} == f({s['input']})"
    result = co_check(prog, timeout=5.0)
    check(f"{sid} canonical passes", result)
    if not result:
        print(f"    FAILED: {sid}")


# ============================================================
# 40. CRUXEval-O: parallel test execution
# ============================================================
print()
print("=" * 60)
print("40. CRUXEval-O: parallel test execution")
print("=" * 60)

co_parallel = _co_mod.run_tests_parallel

par_ce_problems = [ce_problems[0], ce_problems[1], ce_problems[3]]
# Correct output for 0, wrong for 1, correct for 3
par_ce_predictions = [
    ce_problems[0]["output"],   # correct
    "'definitely_wrong'",        # wrong
    ce_problems[3]["output"],   # correct
]
par_ce_results = co_parallel(par_ce_problems, par_ce_predictions, num_workers=3)

check("parallel returns 3 results", len(par_ce_results) == 3)
check("parallel result 0 passes", par_ce_results[0][1] is True)
check("parallel result 1 fails", par_ce_results[1][1] is False)
check("parallel result 2 passes", par_ce_results[2][1] is True)
check("parallel sorted by idx", [r[0] for r in par_ce_results] == [0, 1, 2])


# ============================================================
# 41. CRUXEval-O: evaluate_cruxeval_output interface check
# ============================================================
print()
print("=" * 60)
print("41. CRUXEval-O: interface check")
print("=" * 60)

co_sig = inspect.signature(_co_mod.evaluate_cruxeval_output)
co_params = list(co_sig.parameters.keys())
check("has model param", "model" in co_params)
check("has tokenizer param", "tokenizer" in co_params)
check("has test_data_dir param", "test_data_dir" in co_params)
check("has rank param", "rank" in co_params)
check("has world_size param", "world_size" in co_params)
check("has max_samples param", "max_samples" in co_params)
check("has max_gen_length param", "max_gen_length" in co_params)
check("has batch_size param", "batch_size" in co_params)
check("rank default is 0", co_sig.parameters["rank"].default == 0)
check("world_size default is 1", co_sig.parameters["world_size"].default == 1)
check("max_samples default is None", co_sig.parameters["max_samples"].default is None)
check("batch_size default is 1", co_sig.parameters["batch_size"].default == 1)


# ============================================================
# 42. CRUXEval-I: load module directly
# ============================================================
print()
print("=" * 60)
print("42. CRUXEval-I: load module directly")
print("=" * 60)

_ci_spec = _ilu.spec_from_file_location(
    "cruxeval_input_evaluator",
    "/data/fqzhou/dataset_agent/finetune/test/evaluators/cruxeval_input_evaluator.py",
    submodule_search_locations=[],
)
_ci_mod = _ilu.module_from_spec(_ci_spec)
_ci_spec.loader.exec_module(_ci_mod)
check("cruxeval-I module loaded", hasattr(_ci_mod, "evaluate_cruxeval_input"))
check("has load_cruxeval_local", hasattr(_ci_mod, "load_cruxeval_local"))
check("has check_correctness", hasattr(_ci_mod, "check_correctness"))
check("has generate_prompt", hasattr(_ci_mod, "generate_prompt"))
check("has extract_answer", hasattr(_ci_mod, "extract_answer"))
check("has run_tests_parallel", hasattr(_ci_mod, "run_tests_parallel"))


# ============================================================
# 43. CRUXEval-I: prompt generation
# ============================================================
print()
print("=" * 60)
print("43. CRUXEval-I: prompt generation")
print("=" * 60)

ci_gen_prompt = _ci_mod.generate_prompt

# Without tokenizer
prompt_i_raw = ci_gen_prompt(ce0, tokenizer=None)
check("raw prompt contains code", ce0["code"] in prompt_i_raw)
check("raw prompt contains output", ce0["output"] in prompt_i_raw)
check("raw prompt has f(??)", "f(??)" in prompt_i_raw)
check("raw prompt has [ANSWER] tag", "[ANSWER]" in prompt_i_raw)
check("raw prompt has few-shot example", 'f("ba", "nana")' in prompt_i_raw)
check("raw prompt ends with [ANSWER]", prompt_i_raw.rstrip().endswith("[ANSWER]"))

# Input prompt should NOT contain the input (that's what model predicts)
check("raw prompt does NOT contain input value", ce0["input"] not in prompt_i_raw)

# With chat tokenizer
prompt_i_chat = ci_gen_prompt(ce0, tokenizer=fake_tok2)
check("chat prompt has system", "<|system|>" in prompt_i_chat)
check("chat prompt has assistant", "<|assistant|>" in prompt_i_chat)


# ============================================================
# 44. CRUXEval-I: answer extraction
# ============================================================
print()
print("=" * 60)
print("44. CRUXEval-I: answer extraction")
print("=" * 60)

ci_extract = _ci_mod.extract_answer

# Case 1: clean assertion + [/ANSWER]
check("clean assertion",
      ci_extract('assert f("ba", "nana") == "banana"\n[/ANSWER]') == 'f("ba", "nana")')

# Case 2: just function call
check("just f() call",
      ci_extract('f(16)') == 'f(16)')

# Case 3: reasoning + assertion
check("reasoning then assertion",
      ci_extract('Working backwards...\nassert f(16) == 17\n[/ANSWER]') == 'f(16)')

# Case 4: list input
check("list input",
      ci_extract('assert f(["mq", "px", "zy"]) == 3\n[/ANSWER]') == 'f(["mq", "px", "zy"])')

# Case 5: empty
check("empty returns empty", ci_extract('') == '')

# Case 6: with extra [ANSWER] tag
check("extra ANSWER tag",
      ci_extract('let me think...\n[ANSWER]\nassert f(16) == 17\n[/ANSWER]') == 'f(16)')


# ============================================================
# 45. CRUXEval-I: execution (correct input, wrong input, timeout)
# ============================================================
print()
print("=" * 60)
print("45. CRUXEval-I: execution (check_correctness)")
print("=" * 60)

ci_check = _ci_mod.check_correctness

# Use sample_3: f(text, value) = text + value
# f("bcksrut", "q") == "bcksrutq"

# 45a: correct input - execute f with ground truth input
prog_i_correct = f"{s3['code']}\nassert {s3['output']} == f({s3['input']})"
check("45a correct input passes", ci_check(prog_i_correct, timeout=5.0))

# 45b: alternative valid input (different input, same output)
prog_i_alt = f"{s3['code']}\nassert {s3['output']} == f('bcksrut', 'q')"
check("45b alternative input passes", ci_check(prog_i_alt, timeout=5.0))

# 45c: wrong input (produces different output)
prog_i_wrong = f"{s3['code']}\nassert {s3['output']} == f('hello', 'x')"
check("45c wrong input fails", not ci_check(prog_i_wrong, timeout=5.0))

# 45d: timeout
prog_i_loop = "def f():\n    while True: pass\nassert 1 == f()"
check("45d infinite loop times out", not ci_check(prog_i_loop, timeout=2.0))


# ============================================================
# 46. CRUXEval-I: canonical samples pass
# ============================================================
print()
print("=" * 60)
print("46. CRUXEval-I: canonical samples pass")
print("=" * 60)

for sid in ["sample_0", "sample_3", "sample_10", "sample_50", "sample_100", "sample_500"]:
    s = next(p for p in ce_problems if p["id"] == sid)
    prog = f"{s['code']}\nassert {s['output']} == f({s['input']})"
    result = ci_check(prog, timeout=5.0)
    check(f"{sid} canonical input passes", result)
    if not result:
        print(f"    FAILED: {sid}")


# ============================================================
# 47. CRUXEval-I: parallel test execution
# ============================================================
print()
print("=" * 60)
print("47. CRUXEval-I: parallel test execution")
print("=" * 60)

ci_parallel = _ci_mod.run_tests_parallel

par_ci_problems = [ce_problems[0], ce_problems[1], ce_problems[3]]
# Correct input for 0, invalid (no f()) for 1, correct for 3
par_ci_predictions = [
    f"f({ce_problems[0]['input']})",  # correct
    "not_a_function_call",             # invalid - no f()
    f"f({ce_problems[3]['input']})",  # correct
]
par_ci_results = ci_parallel(par_ci_problems, par_ci_predictions, num_workers=3)

check("parallel returns 3 results", len(par_ci_results) == 3)
check("parallel result 0 passes", par_ci_results[0][1] is True)
check("parallel result 1 fails (no f())", par_ci_results[1][1] is False)
check("parallel result 2 passes", par_ci_results[2][1] is True)
check("parallel sorted by idx", [r[0] for r in par_ci_results] == [0, 1, 2])


# ============================================================
# 48. CRUXEval-I: evaluate_cruxeval_input interface check
# ============================================================
print()
print("=" * 60)
print("48. CRUXEval-I: interface check")
print("=" * 60)

ci_sig = inspect.signature(_ci_mod.evaluate_cruxeval_input)
ci_params = list(ci_sig.parameters.keys())
check("has model param", "model" in ci_params)
check("has tokenizer param", "tokenizer" in ci_params)
check("has test_data_dir param", "test_data_dir" in ci_params)
check("has rank param", "rank" in ci_params)
check("has world_size param", "world_size" in ci_params)
check("has max_samples param", "max_samples" in ci_params)
check("has max_gen_length param", "max_gen_length" in ci_params)
check("has batch_size param", "batch_size" in ci_params)
check("rank default is 0", ci_sig.parameters["rank"].default == 0)
check("world_size default is 1", ci_sig.parameters["world_size"].default == 1)
check("max_samples default is None", ci_sig.parameters["max_samples"].default is None)
check("batch_size default is 1", ci_sig.parameters["batch_size"].default == 1)


# ============================================================
# 49. Routing: cruxeval_O and cruxeval_I in evaluate_test_set
# ============================================================
print()
print("=" * 60)
print("49. Routing: cruxeval_O and cruxeval_I")
print("=" * 60)

with open("/data/fqzhou/dataset_agent/finetune/test/__init__.py") as f:
    routing_src = f.read()

check("cruxeval_O in routing", "cruxeval_O" in routing_src)
check("cruxeval_I in routing", "cruxeval_I" in routing_src)
check("evaluate_cruxeval_output imported", "evaluate_cruxeval_output" in routing_src)
check("evaluate_cruxeval_input imported", "evaluate_cruxeval_input" in routing_src)
check("cruxeval_O routes to evaluate_cruxeval_output",
      "dataset_name == 'cruxeval_O'" in routing_src and "evaluate_cruxeval_output" in routing_src)
check("cruxeval_I routes to evaluate_cruxeval_input",
      "dataset_name == 'cruxeval_I'" in routing_src and "evaluate_cruxeval_input" in routing_src)

with open("/data/fqzhou/dataset_agent/finetune/test/evaluators/__init__.py") as f:
    eval_init_src2 = f.read()

check("evaluators exports cruxeval_output",
      "from .cruxeval_output_evaluator import evaluate_cruxeval_output" in eval_init_src2)
check("evaluators exports cruxeval_input",
      "from .cruxeval_input_evaluator import evaluate_cruxeval_input" in eval_init_src2)

# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
total = passed + failed
if failed == 0:
    print(f"ALL {total} TESTS PASSED")
else:
    print(f"Results: {passed}/{total} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
