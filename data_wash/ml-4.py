import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os  # 用于文件夹创建和路径拼接
import joblib # 导入并行计算库，用于模型并行训练
from sklearn.model_selection import train_test_split, cross_val_score# 导入模型选择模块，用于数据拆分和交叉验证
from sklearn.linear_model import LinearRegression# 导入线性回归模型，用于基准模型
from sklearn.ensemble import RandomForestRegressor# 导入随机森林回归模型，用于核心模型评估
import xgboost as xgb# 导入XGBoost库，用于核心模型评估和特征重要性
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score# 导入模型评估指标，用于模型评估

# ===================== 基础配置（核心：中文乱码 + 双文件夹路径）=====================
# 解决matplotlib中文显示问题（兼容Windows系统）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文显示字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['savefig.dpi'] = 300  # 保存图片高清分辨率
plt.rcParams['figure.dpi'] = 100  # 画布显示分辨率

# 定义图片保存路径，自动创建image文件夹（无则建，有则跳过）
img_save_dir = r'C:\Users\hk156\Desktop\python_Pa\data_wash\image'
os.makedirs(img_save_dir, exist_ok=True)
# 定义pkl文件保存路径，自动创建pklpage文件夹（核心修改）
#pkl用于数据持久化，像保存机器学习模型、中间结果等，避免重复计算；还能跨平台共享数据，在机器学习和深度学习里常用来保存模型或预处理步骤
pkl_save_dir = r'C:\Users\hk156\Desktop\python_Pa\data_wash\pklpage'
os.makedirs(pkl_save_dir, exist_ok=True)

print(f"图片将保存至：{img_save_dir}")
print(f"PKL模型文件将保存至：{pkl_save_dir}")

# 1. 加载特征数据
df = pd.read_csv('featured_house_data.csv')# 读取特征数据，包含所有特征列和目标列（price）
# 提取特征列（排除目标列）
feature_cols = [col for col in df.columns if col != 'price']
target_col = 'price'

# 2. 拆分训练集/测试集（7:3）
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # random_state固定，保证结果可复现
)
print(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

# 3. 模型定义（基准+核心）
models = {
    '线性回归（基准）': LinearRegression(),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    # 【核心修改】显式指定base_score=0.5，SHAP可正常解析，从源头解决报错
    'XGBoost（核心）': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, base_score=0.5)
}

# 4. 模型训练与评估
results = {}
print("\n===== 模型评估结果 =====")
for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    # 交叉验证（提升可信度）
    cv_r2 = cross_val_score(model, X, y, cv=5, scoring='r2').mean()
    # 保存结果
    results[name] = {
        'MAE': round(mae, 2),# 平均绝对误差（MAE），评估模型预测值与真实值的平均绝对差,越小越好
        'RMSE': round(rmse, 2),# 均方根误差（RMSE），评估模型预测值与真实值的均方根误差,越小越好平方和开根号
        'R²': round(r2, 3),# 决定系数（R²），评估模型预测值与真实值相关程度，越接近1越好，0表示无相关
        #将数据集分成 5 份，每次用其中 4 份训练模型，1 份测试，重复 5 次，使每个数据都有机会做训练和测试，
        #以此评估模型泛化能力，减少因数据划分带来的偏差。
        # 1-（y-pred）^2/（y-mean(y)）^2，越接近1越好，0表示无相关
        '5折交叉验证R²': round(cv_r2, 3)# 5折交叉验证R²，评估模型在不同数据子集上的泛化能力，越接近1越好，0表示无相关
    }
    # 打印结果
    print(f"\n{name}：")
    print(f"  MAE（平均绝对误差）：{mae:.2f} 元/㎡")
    print(f"  RMSE（均方根误差）：{rmse:.2f} 元/㎡")
    print(f"  R²（决定系数）：{r2:.3f}")
    print(f"  5折交叉验证R²：{cv_r2:.3f}")

# 5. 可视化：模型效果对比（R²）
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
r2_scores = [results[name]['R²'] for name in model_names]
cv_r2_scores = [results[name]['5折交叉验证R²'] for name in model_names]
x = np.arange(len(model_names))
width = 0.35

# 绘制柱状图
plt.bar(x - width/2, r2_scores, width, label='测试集R²', color='lightblue')
plt.bar(x + width/2, cv_r2_scores, width, label='5折交叉验证R²', color='lightgreen')

# 图表标签（中文正常显示）
plt.title('各模型R²效果对比', fontsize=14, pad=10)
plt.xlabel('模型', fontsize=12)
plt.ylabel('R²（越接近1越好）', fontsize=12)
plt.xticks(x, model_names, rotation=15, fontsize=10)
plt.legend(fontsize=10)
plt.tight_layout()  # 自动调整布局，避免标签截断

# 保存图片到指定image文件夹
plt.savefig(os.path.join(img_save_dir, 'model_comparison.png'), bbox_inches='tight')
plt.show()
print(f"\n模型对比图已保存至：{os.path.join(img_save_dir, 'model_comparison.png')}")

# 6. 保存最优模型（XGBoost）和特征列表到指定pklpage文件夹
#极限梯度提升算法（XGBoost）是一种基于决策树的梯度提升框架，用于解决分类和回归问题
best_model = models['XGBoost（核心）']
# 拼接pkl文件完整路径，避免保存到默认目录
model_pkl_path = os.path.join(pkl_save_dir, 'xgboost_house_price_model.pkl')
feature_pkl_path = os.path.join(pkl_save_dir, 'feature_cols.pkl')
# joblib主要用于保存模型和特征列到.pkl文件
joblib.dump(best_model, model_pkl_path)
joblib.dump(feature_cols, feature_pkl_path)

print(f"最优模型（XGBoost）已保存到：{model_pkl_path}")
print(f"特征列表已保存到：{feature_pkl_path}")