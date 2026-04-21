import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats  # 用于核密度曲线
import warnings
import os  # 新增：用于创建文件夹
from tools import clean_price, clean_area
warnings.filterwarnings('ignore')

# ===================== 基础配置 + 文件夹创建 =====================
# 设置中文字体（解决乱码），兼容多系统
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.dpi'] = 100  # 全局画布分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片高清分辨率

# 定义图片保存路径，并自动创建文件夹（无则建，有则跳过）
save_path = r'C:\Users\hk156\Desktop\python_Pa\data_wash\image'
os.makedirs(save_path, exist_ok=True)
print(f"图片将保存至：{save_path}")

# ===================== 1. 加载原始数据 =====================
df = pd.read_csv(r'C:\Users\hk156\Desktop\python_Pa\data_wash\cleaned_house_data.csv')
print(f"【原始数据】总行数：{len(df)}")

# 注：clean_price和clean_area函数已移至tools.py模块

# ===================== 3. 执行清洗+过滤极端值 =====================
# 字段清洗
df['price'] = df['price'].apply(clean_price)
df['area'] = df['area'].apply(clean_area)

# 过滤无效数据：价格/面积非空 + 面积30-500㎡ + 房价0-5万/㎡（过滤极端高价，聚焦主流市场）
df_valid = df.dropna(subset=['price', 'area'])
df_valid = df_valid[(df_valid['area'] >= 30) & (df_valid['area'] <= 500)]
df_valid = df_valid[(df_valid['price'] > 0) & (df_valid['price'] <= 50000)]

print(f"【清洗后】有效行数：{len(df_valid)}（过滤了极端值/无效值）")
if len(df_valid) == 0:
    print("⚠️  无有效数据，自动加载测试数据演示！")
    # 测试数据兜底（模拟合肥区域房价）
    df_valid = pd.DataFrame({
        'price': [27000, 29000, 30000, 23500, 17000, 16000, 21000, 24800, 32000, 28000],
        'area': [(143+248)/2, (141+221)/2, (143+223)/2, (180+230)/2, (105+167)/2, 
                 (115+188)/2, (102+143)/2, (149+188)/2, (120+200)/2, (90+150)/2],
        'district': ['滨湖', '滨湖', '经开', '包河', '包河', '肥西', '肥西', '蜀山', '政务', '政务']
    })

# ===================== 4. 核心统计信息输出 =====================
print("\n===== 核心字段统计描述（面积/房价） =====")
print(df_valid[['area', 'price']].describe().round(2))

# ===================== 5. 可视化1：房价分布（【优化版箱型图】+ 核密度直方图） =====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# 左：箱型图【重点优化】- 加宽箱体、突出异常值、添加数值标注（中位数/四分位数）
box = ax1.boxplot(df_valid['price'], patch_artist=True, 
                  boxprops={'facecolor': '#87CEEB', 'alpha': 0.8, 'linewidth': 1.5},  # 加宽箱体边框、提高透明度
                  whiskerprops={'color': '#4682B4', 'linewidth': 1.5, 'linestyle': '--'},  # 虚线须线更醒目
                  medianprops={'color': '#DC143C', 'linewidth': 2.5},  # 加粗中位数线
                  flierprops={'marker': 'o', 'markerfacecolor': '#FF6347', 'markersize': 4, 'alpha': 0.6},  # 异常值红色圆点
                  widths=0.6)  # 加宽箱体，视觉更舒适

# 箱型图添加关键数值标注（中位数、上下四分位数）
price_stats = df_valid['price'].describe()
q1, med, q3 = price_stats['25%'], price_stats['50%'], price_stats['75%']
ax1.text(1.1, med, f'中位数：{med:.0f}', ha='left', va='center', fontsize=10, color='#DC143C', fontweight='bold')
ax1.text(1.1, q1, f'下四分位：{q1:.0f}', ha='left', va='center', fontsize=9, color='#4682B4')
ax1.text(1.1, q3, f'上四分位：{q3:.0f}', ha='left', va='center', fontsize=9, color='#4682B4')

ax1.set_title('房价单价分布箱型图（元/㎡）', fontsize=12, pad=10, fontweight='bold')
ax1.set_ylabel('价格（元/㎡）', fontsize=10)
ax1.set_xlim(0.5, 1.8)  # 调整X轴范围，给数值标注留空间
ax1.grid(axis='y', alpha=0.3, linewidth=1)

# 右：直方图+核密度曲线（保持不变，清晰看价格集中区间）
ax2.hist(df_valid['price'], bins=25, color='#98FB98', edgecolor='black', alpha=0.7, density=True, label='直方图')
# 核密度曲线
kde = stats.gaussian_kde(df_valid['price'])
x_range = np.linspace(df_valid['price'].min(), df_valid['price'].max(), 200)
ax2.plot(x_range, kde(x_range), color='#DC143C', linewidth=2, label='核密度曲线')
ax2.set_title('房价单价分布（直方图+核密度）', fontsize=12, pad=10, fontweight='bold')
ax2.set_xlabel('价格（元/㎡）', fontsize=10)
ax2.set_ylabel('密度', fontsize=10)
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(alpha=0.3, linewidth=1)

plt.tight_layout()
# 保存到指定image文件夹
plt.savefig(os.path.join(save_path, 'price_distribution_optimize.png'), bbox_inches='tight')
plt.show()

# ===================== 6. 可视化2：面积vs价格散点图（单一颜色、无区域配色/图例） =====================
plt.figure(figsize=(10, 6))
# 原版样式：橙色散点、透明度0.6、无区域区分、无图例
plt.scatter(df_valid['area'], df_valid['price'], alpha=0.6, color='orange', s=30)

plt.title('面积 vs 房价单价散点图', fontsize=14, pad=15)
plt.xlabel('面积（㎡，范围中间值）', fontsize=12)
plt.ylabel('价格（元/㎡）', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
# 保存到指定image文件夹
plt.savefig(os.path.join(save_path, 'area_price_corr_optimize.png'), bbox_inches='tight')
plt.show()

# ===================== 7. 可视化3：区域房价Top20（横向柱状图，彻底解决文字重叠） =====================
# 按区域分组统计，保留样本数≥10的区域（保证统计意义）
district_price = df_valid.groupby('district')['price'].agg(['mean', 'median', 'count']).round(2)
district_price = district_price[district_price['count'] >= 10].sort_values('mean', ascending=False)
# 取均价Top20区域，若不足20则取全部
top_n = min(20, len(district_price))
district_price_top20 = district_price.head(top_n)

print(f"\n===== 各区域房价统计（样本数≥10，共{len(district_price)}个区域，展示Top{top_n}） =====")
print(district_price_top20)

# 绘制横向柱状图
plt.figure(figsize=(12, 8))
bars = plt.barh(range(len(district_price_top20)), district_price_top20['mean'], 
                color=plt.cm.Reds(np.linspace(0.4, 0.9, len(district_price_top20))))
# 倒序显示（高价在顶部）
plt.yticks(range(len(district_price_top20)), district_price_top20.index, fontsize=10)
plt.gca().invert_yaxis()

# 给柱子添加数值标签
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width + 200, bar.get_y() + bar.get_height()/2, f'{width:.0f}', 
             ha='left', va='center', fontsize=9)

plt.title(f'各区域房价均价Top{top_n}（样本数≥10）', fontsize=14, pad=15, fontweight='bold')
plt.xlabel('均价（元/㎡）', fontsize=12)
plt.ylabel('区域', fontsize=12)
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
# 保存到指定image文件夹
plt.savefig(os.path.join(save_path, 'district_price_top20_optimize.png'), bbox_inches='tight')
plt.show()

# ===================== 8. 数值字段相关性分析 =====================
numeric_cols = df_valid.select_dtypes(include=['int64', 'float64']).columns
if 'price' in numeric_cols and len(numeric_cols) >= 2:
    corr = df_valid[numeric_cols].corr()
    print("\n===== 数值字段相关性（与房价的相关性排序） =====")
    print(corr['price'].sort_values(ascending=False).round(4))
else:
    print("\n===== 数值字段相关性 =====")
    print("有效数值字段不足，无法进行相关性分析")

print(f"\n 所有分析完成！3张高清图表已保存至：{save_path}")