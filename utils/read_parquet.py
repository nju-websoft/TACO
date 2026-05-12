import pandas as pd
import pyarrow.parquet as pq

# 1. 替换成你的 parquet 文件路径
parquet_file = "dataset/mmlu_pro/test.parquet"

# ----------------------
# 方法1：快速查看（推荐日常使用）
# ----------------------
print("===== 1. 快速查看 Parquet 结构 =====")
df = pd.read_parquet(parquet_file)
# 打印完整结构：列名 + 数据类型 + 行列数 + 前5行预览
print(df.info())
print("\n前5行数据：")
print(df.head())

# ----------------------
# 方法2：专业元数据查看（查看底层信息）
# ----------------------
print("\n===== 2. Parquet 底层元数据 =====")
parquet = pq.ParquetFile(parquet_file)

# 1. 元数据总览
print("文件元数据：")
print(parquet.metadata)

print("\n===== 3. 列信息（Schema）=====" )
# 2. 打印每一列的名称 + 类型 + 是否可为空
for i, field in enumerate(parquet.schema.names):
    dtype = parquet.schema.types[i]
    nullable = parquet.schema.nullables[i]
    print(f"列 {i+1}: {field} | 类型: {dtype} | 可为空: {nullable}")

# 3. 行列统计
print(f"\n===== 4. 数据尺寸 =====")
print(f"总行数: {parquet.metadata.num_rows}")
print(f"总列数: {len(parquet.schema.names)}")
print(f"数据文件行数组(Row Groups): {parquet.num_row_groups}")