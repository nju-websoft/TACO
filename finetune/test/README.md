# Test Evaluator Module

模块化的测试评测框架,支持多种数据集类型的评测。

## 目录结构

```
test/
├── __init__.py                 # 统一入口,提供 evaluate_test_set()
├── evaluators/                 # 各类评测器
│   ├── __init__.py
│   ├── math_evaluator.py      # 数学推理评测 (GSM8K, MATH, Minerva等)
│   └── mmlu_evaluator.py      # 多选题评测 (MMLU, MMLU-Pro)
└── utils/                      # 工具模块
    ├── __init__.py
    ├── answer_utils.py         # 答案提取和清理
    ├── data_utils.py           # 数据加载
    └── verify_utils.py         # 数学验证
```

## 使用方法

### 统一接口 (推荐)

```python
from test import evaluate_test_set

# 自动根据数据集名称选择评测器
accuracy, correct, total = evaluate_test_set(
    model=model,
    tokenizer=tokenizer,
    test_data_dir="/path/to/gsm8k",  # 或 mmlu_pro
    rank=0,
    world_size=1,
    max_samples=None,
    max_gen_length=1024,
    batch_size=16
)
```

### 直接使用特定评测器

```python
from test.evaluators import evaluate_exact_match, evaluate_mmlu_choice

# 数学推理评测
accuracy, correct, total = evaluate_exact_match(
    model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size
)

# MMLU评测
accuracy, correct, total = evaluate_mmlu_choice(
    model, tokenizer, test_data_dir, rank, world_size, max_samples, max_gen_length, batch_size
)
```

## 支持的数据集

- **数学推理**: gsm8k, math, minerva, numina_cot (使用 boxed 答案格式)
- **多选题**: mmlu, mmlu_pro (使用字母选项 A-J)
- **代码生成**: LiveCodeBench (代码生成评测)


## 代码生成评测

```python
from finetune.test.evaluators import evaluate_code_generation

pass_at_1, correct, total = evaluate_code_generation(
    model, tokenizer, batch_size=1, rank=0, world_size=1
)
```

详见 [LIVECODEBENCH.md](LIVECODEBENCH.md)

## LiveCodeBench 集成

支持代码生成任务评测,详见 [LIVECODEBENCH.md](LIVECODEBENCH.md)

```python
from finetune.test.evaluators import evaluate_livecodebench

results = evaluate_livecodebench(
    model=model,
    tokenizer=tokenizer,
    scenario='codegeneration',
    batch_size=4
)
print(f"pass@1: {results['pass@1']:.2%}")
```

## 向后兼容

原有的 `test_evaluator_distributed.py` 接口保持不变,可以直接替换导入:

```python
# 旧方式
from test_evaluator_distributed import evaluate_test_set

# 新方式 (完全兼容)
from test import evaluate_test_set
```

## 扩展新的评测器

在 `test/evaluators/` 下添加新的评测器文件,然后在 `test/__init__.py` 中注册即可。
