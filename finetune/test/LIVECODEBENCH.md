# LiveCodeBench 集成使用说明

LiveCodeBench 代码生成评测已集成到 test 模块。

## 使用方法

### 基本调用

```python
from finetune.test.evaluators import evaluate_code_generation

# 评测代码生成
pass_at_1, correct, total = evaluate_code_generation(
    model=model,
    tokenizer=tokenizer,
    release_version='release_latest',
    rank=0,
    world_size=1,
    max_samples=None,
    max_gen_length=2048,
    batch_size=1
)

print(f"pass@1: {pass_at_1:.2%}")
```

### 在 train.py 中使用

与数学评测方式完全一致:

```python
from finetune.test.evaluators import evaluate_code_generation

# 在训练循环中
if step % eval_interval == 0:
    pass_at_1, correct, total = evaluate_code_generation(
        model=model,
        tokenizer=tokenizer,
        batch_size=1,
        rank=local_rank,
        world_size=world_size
    )
    
    if local_rank == 0:
        print(f"Code generation pass@1: {pass_at_1:.2%}")
```

### 参数说明

- `model`: 待评测模型
- `tokenizer`: tokenizer
- `release_version`: 数据集版本 (默认: 'release_latest')
- `rank`: 分布式 rank
- `world_size`: 分布式总进程数
- `max_samples`: 最大评测样本数 (None 表示全部)
- `max_gen_length`: 最大生成长度
- `batch_size`: 批处理大小

### 返回值

返回元组 `(pass_at_1, correct, total)`:
- `pass_at_1`: Pass@1 准确率 (0-1 之间的浮点数)
- `correct`: 通过的问题数
- `total`: 总问题数

## 实现说明

- 复用 LiveCodeBench 的核心评测逻辑 (benchmarks, evaluation, prompts, utils)
- 只支持本地模型评测,不需要 API runner
- 与数学/MMLU 评测器保持一致的接口
