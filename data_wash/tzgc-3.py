import pandas as pd# 导入Pandas库，用于数据处理
import numpy as np# 导入NumPy库，用于数值计算
import re# 导入正则表达式库，用于字符串匹配
from sklearn.preprocessing import LabelEncoder, StandardScaler# 导入标签编码器和标准缩放器
from sklearn.impute import SimpleImputer# 导入简单缺失值填充器，用于填充数值和类别字段的缺失值
from tools import clean_price, clean_area, parse_rooms, remove_outliers# 导入自定义的清洗函数，用于处理价格、面积、户型和异常值剔除

# 1. 加载数据
df = pd.read_csv('cleaned_house_data.csv')

# 清洗price列并转换为数值型
df['price'] = df['price'].apply(clean_price)
print(f"price列清洗后，缺失值数量：{df['price'].isna().sum()}")

# 清洗area列并转换为数值型
df['area'] = df['area'].apply(clean_area)
print(f"area列清洗后，缺失值数量：{df['area'].isna().sum()}")

# 2. 数据清洗（缺失值+异常值）
# 2.1 缺失值处理（用均值/众数填充）
# 数值字段：用均值填充（新增price列的缺失值处理）
num_imputer = SimpleImputer(strategy='mean')# 创建简单缺失值填充器，用均值填充数值字段
df['area'] = num_imputer.fit_transform(df[['area']]).ravel()# 填充area缺失值
df['price'] = num_imputer.fit_transform(df[['price']]).ravel()  # 填充price缺失值

# 类别字段：用众数填充
cat_imputer = SimpleImputer(strategy='most_frequent')# 创建简单缺失值填充器，用众数填充类别字段
df['district'] = cat_imputer.fit_transform(df[['district']]).ravel()# 填充district缺失值
df['rooms'] = cat_imputer.fit_transform(df[['rooms']]).ravel()# 填充rooms缺失值

# 2.2 异常值处理（价格/面积：用IQR法剔除）
df = remove_outliers(df, 'price')# 去除价格异常值
df = remove_outliers(df, 'area')# 去除面积异常值
print(f"异常值处理后数据量：{len(df)} 条")

# 3. 类别特征编码（区域/户型）
# 3.1 区域：标签编码（因为区域数量多，独热编码会维度爆炸）
le_district = LabelEncoder()# 创建标签编码器，用于区域标签编码
df['district_encoded'] = le_district.fit_transform(df['district'])# 对district列进行标签编码
# 3.2 户型：先标准化（比如「四居」→4，「三居」→3），再编码
df['rooms_num'] = df['rooms'].apply(parse_rooms)# 对rooms列进行解析，提取房间数量

# 4. 高阶特征构造（体现特征工程思维）
# 4.1 单价相关：面积/户型数=单户型平均面积
df['avg_area_per_room'] = df['area'] / df['rooms_num'].replace(0, 1)  # 避免除0
# 4.2 区域统计特征：各区域均价（作为特征）
district_mean_price = df.groupby('district')['price'].mean().to_dict()
# 计算每个区域的平均价格，用于后续特征工程
df['district_mean_price'] = df['district'].map(district_mean_price)
# 4.3 面积分箱：把面积分成区间（如0-90/90-120/120+）
df['area_bin'] = pd.cut(df['area'], bins=[0, 90, 120, 200, 500], 
                        labels=[0, 1, 2, 3], include_lowest=True)

# 5. 特征筛选（只保留用于建模的特征）
feature_cols = [
    'area', 'rooms_num', 'district_encoded', 
    'avg_area_per_room', 'district_mean_price', 'area_bin'
]
target_col = 'price'

# 6. 数值特征标准化（提升模型效果）
scaler = StandardScaler()# 创建标准缩放器，用于数值特征标准化
scale_cols = ['area', 'rooms_num', 'avg_area_per_room', 'district_mean_price']
# 修复：StandardScaler返回二维数组，无需ravel（Pandas可直接赋值）
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 7. 保存处理后的特征数据（供后续建模）
df[feature_cols + [target_col]].to_csv('featured_house_data.csv', index=False)
print("特征工程完成，保存到 featured_house_data.csv")
print(f"最终特征列表：{feature_cols}")
print(f"特征数据形状：{df[feature_cols].shape}")