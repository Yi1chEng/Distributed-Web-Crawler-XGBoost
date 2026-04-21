import pandas as pd
import numpy as np

# 1. 加载原始数据，不做任何清洗
df = pd.read_csv(r'C:\Users\hk156\Desktop\python_Pa\data_wash\cleaned_house_data.csv')

# 2. 打印核心诊断信息
print("===== 原始数据诊断 =====")
print(f"原始总行数：{len(df)}")
print("\n1. 字段列表：", df.columns.tolist())
print("\n2. price字段前10行值：")
print(df['price'].head(10).tolist())
print("\n3. price字段数据类型：", df['price'].dtype)
print("\n4. district字段前10行值：")
print(df['district'].head(10).tolist())
print("\n5. area字段前10行值：")
print(df['area'].head(10).tolist())

# 6. 统计各字段缺失值
print("\n6. 各字段缺失值数量：")
print(df[['price', 'area', 'district']].isnull().sum())

# 7. 检查price字段的异常格式
print("\n7. price字段非数值样本（前5个）：")
non_num_price = df[~df['price'].astype(str).str.match(r'^[\d\.万/㎡元]*$', na=False)]
print(non_num_price['price'].head(5).tolist())