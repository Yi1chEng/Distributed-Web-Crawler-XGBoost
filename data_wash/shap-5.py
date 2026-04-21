# 导入必要的库
import pandas as pd  # 数据处理库
import numpy as np  # 数值计算库
import shap  # 模型解释库解释机器学习模型中每个特征对预测结果的贡献，帮助理解模型决策过程。
import matplotlib.pyplot as plt  # 绘图库
import joblib  # 模型保存和加载库
import os  # 文件系统操作库
import re  # 正则表达式库
import xgboost as xgb  # XGBoost模型库

# ===================== 基础配置（解决中文乱码+路径）=====================
# 设置Matplotlib中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 设置保存图片的DPI，提高图片清晰度
plt.rcParams['savefig.dpi'] = 300

# 定义路径
pkl_dir = r'C:\Users\hk156\Desktop\python_Pa\data_wash\pklpage'  # 模型和特征列保存路径
img_dir = r'C:\Users\hk156\Desktop\python_Pa\data_wash\image'  # 图片保存路径
# 创建图片保存目录（如果不存在）
os.makedirs(img_dir, exist_ok=True)

# ===================== 核心方案：绕过SHAP原生XGB解析，用通用预测包装器 =====================
class XGBPredictWrapper:
    """XGBoost预测包装器类，让SHAP无需解析模型底层参数
    
    这个包装器类解决了SHAP直接调用XGBoost模型时可能遇到的参数解析问题，
    通过封装模型的predict方法，使SHAP能够正确调用预测函数。
    """
    def __init__(self, model):
        """初始化包装器
        Args:
            model: 训练好的XGBoost模型
        """
        self.model = model
    
    def __call__(self, X):
        """被SHAP调用的方法
        Args:
            X: 输入特征数据
        Returns:
            模型预测结果
        """
        # 确保输入是numpy数组，避免类型错误
        if not isinstance(X, np.ndarray):#表示N 维数组对象
            X = np.array(X)
        # 调用模型的predict方法，设置validate_features=False避免特征验证错误
        return self.model.predict(X, validate_features=False)#跳过输入数据特征与模型训练时特征的一致性检查

# ===================== 加载模型/数据 =====================
# 加载训练好的XGBoost模型
best_model = joblib.load(os.path.join(pkl_dir, 'xgboost_house_price_model.pkl'))
# 加载特征列名称
feature_cols = joblib.load(os.path.join(pkl_dir, 'feature_cols.pkl'))
# 加载处理后的特征数据
print(f"正在加载数据...")
df = pd.read_csv('featured_house_data.csv')
# 提取特征数据
X = df[feature_cols]
print(f"数据加载完成，共有 {len(X)} 条记录")

# 大数据集采样（提升速度，避免内存溢出）
sample_size = min(1000, len(X))  # 最多采样1000条数据
print(f"正在采样 {sample_size} 条数据...")
X_sample = X.iloc[:sample_size].reset_index(drop=True)#从特征数据中采样sample_size条数据，重置索引
# 生成SHAP的背景数据（用采样数据的前100条，减少计算量）
background_data = shap.sample(X_sample, 100)
print(f"背景数据生成完成，共 {len(background_data)} 条")

# ===================== 初始化SHAP）=====================
# 创建预测包装器实例
predict_wrapper = XGBPredictWrapper(best_model)#创建XGBPredictWrapper实例，包装best_model
# KernelExplainer是SHAP通用解释器，无需解析XGBoost底层参数，完美解决所有属性缺失/解析报错
explainer = shap.KernelExplainer(predict_wrapper, background_data)
print("SHAP解释器初始化完成，开始计算SHAP值...")
# 计算SHAP值（nsamples=50平衡计算速度和精度，避免内存溢出）
shap_values = explainer.shap_values(X_sample, nsamples=50)#计算SHAP值，返回每个样本的SHAP值向量
print("SHAP值计算完成")

# ===================== 可视化+结果输出（无任何修改，中文正常显示）=====================
print("开始生成可视化结果...")
# 可视化1：SHAP汇总图（保存到image文件夹）
plt.figure(figsize=(10, 6))
# 生成SHAP汇总图，显示特征对模型预测的影响
shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, plot_type='dot', show=False)
plt.title('SHAP特征贡献汇总图（影响房价的核心特征）', pad=15, fontsize=14)
plt.tight_layout()
# 保存图片
plt.savefig(os.path.join(img_dir, 'shap_summary.png'), bbox_inches='tight')
print(f"SHAP汇总图已保存到: {os.path.join(img_dir, 'shap_summary.png')}")
plt.show()

# 输出SHAP特征重要性（官方推荐计算方式）
# 计算每个特征的平均绝对SHAP值作为重要性指标
feature_importance = pd.DataFrame({
    '特征名': feature_cols,
    'SHAP重要性': np.abs(shap_values).mean(axis=0)  # 平均绝对SHAP值
}).sort_values('SHAP重要性', ascending=False).round(4)

# 可视化2：SHAP依赖图（选择最重要的特征）
plt.figure(figsize=(10, 5))
# 选择SHAP重要性最高的特征作为核心特征
if not feature_importance.empty:
    core_feature = feature_importance.iloc[0]['特征名']
else:
    core_feature = feature_cols[0]  # fallback
# 生成SHAP依赖图，显示特征值与SHAP值的关系
shap.dependence_plot(
    core_feature, shap_values, X_sample,
    feature_names=feature_cols, show=False
)
plt.title(f'{core_feature} - SHAP依赖图', pad=15, fontsize=14)
plt.tight_layout()
# 保存图片
plt.savefig(os.path.join(img_dir, 'shap_dependence.png'), bbox_inches='tight')
print(f"SHAP依赖图已保存到: {os.path.join(img_dir, 'shap_dependence.png')}")
plt.show()

# 打印结果
print("="*50)
print("SHAP特征重要性排名（从高到低）")
print("="*50)
print(feature_importance)
print("\n" + "="*50)
print("业务解释")
print("="*50)
if not feature_importance.empty:
    top1_feat = feature_importance.iloc[0]['特征名']
    print(f"1. 影响房价的最核心特征：{top1_feat}")
print(f"2. SHAP值为正：特征值越大 → 房价预测值越高")
print(f"3. SHAP值为负：特征值越大 → 房价预测值越低")
print(f"4. 汇总图中特征越靠左，对房价的影响程度越大")
print("\n分析完成！")