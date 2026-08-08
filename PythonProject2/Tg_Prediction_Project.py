# ============================================================
# 项目：基于机器学习的聚合物玻璃化转变温度（Tg）预测
# 作者：韦宏宇
# 日期：2026年8月
# 数据来源：OpenPoly 实验聚合物数据库
# ============================================================

from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')

# 修复 Windows 命令行 GBK 编码无法输出 emoji/中文符号的问题
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 抑制 RDKit 解析个别不合法 PSMILES 时打印的错误日志（这些样本会被自动剔除）
RDLogger.DisableLog('rdApp.error')

# 配置 matplotlib 支持中文显示，避免图表标题/标签乱码
import matplotlib.font_manager as fm
_chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
_avail = {f.name for f in fm.fontManager.ttflist}
_chosen = next((f for f in _chinese_fonts if f in _avail), None)
if _chosen:
    plt.rcParams['font.sans-serif'] = [_chosen]
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示为方块的问题

# ==================== 1. 数据加载 ====================
print("=" * 60)
print("聚合物玻璃化转变温度（Tg）预测模型")
print("=" * 60)

df = pd.read_excel('experiment_polymer_data.xlsx')
df_tg = df[df['Tg_K'].notna()].copy()
print(f"✅ 加载数据完成，共 {len(df_tg)} 个有 Tg 值的样本")

# ==================== 2. 特征工程 ====================
# 使用 RDKit 从 PSMILES（聚合物的分子结构表示）计算分子描述符，
# 替代原先粗糙的字符串计数特征，获得更有物理意义的分子性质

def extract_rdkit_descriptors(smiles):
    """用 RDKit 从 PSMILES 计算分子描述符；解析或计算失败返回 None"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return {
            'MolWt': Descriptors.MolWt(mol),                                   # 分子量
            'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),                 # 重原子数
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),           # 可旋转键数（影响链柔性）
            'NumHDonors': Descriptors.NumHDonors(mol),                         # 氢键供体数
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),                   # 氢键受体数
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),             # 芳香环数（影响链刚性）
            'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),           # 饱和环数
            'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),           # 脂肪环数
            'RingCount': Descriptors.RingCount(mol),                           # 环总数
            'TPSA': Descriptors.TPSA(mol),                                     # 极性表面积
            'MolLogP': Descriptors.MolLogP(mol),                               # 脂水分配系数（疏水性）
            'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),                 # 杂原子数
        }
    except Exception:
        return None

print("正在用 RDKit 计算分子描述符...")
# 同时收集特征与目标，跳过 PSMILES 无法解析的样本
# 先重置索引，确保 iloc 序号与样本位置一致，避免潜在错位
df_tg = df_tg.reset_index(drop=True)
feature_list = []
y_list = []
invalid_count = 0
for idx, s in enumerate(df_tg['PSMILES'].values):
    desc = extract_rdkit_descriptors(s)
    if desc is None:
        invalid_count += 1
        continue
    feature_list.append(desc)
    y_list.append(df_tg.loc[idx, 'Tg_K'])

if invalid_count > 0:
    print(f"⚠️ 有 {invalid_count} 个样本 PSMILES 无法解析，已剔除")

X = pd.DataFrame(feature_list)
y = pd.Series(y_list, name='Tg_K')
print(f"✅ 特征提取完成，共 {X.shape[1]} 个 RDKit 描述符特征，{len(X)} 个有效样本")

# ==================== 3. 划分数据集 ====================
# 先划分出测试集（最终评估用，绝不参与训练过程）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 再从训练集中切出一部分作为验证集（用于早停监控，避免数据泄漏）
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42
)
print(f"✅ 数据集划分：训练集 {len(X_train)} 个，验证集 {len(X_val)} 个，测试集 {len(X_test)} 个")

# ==================== 3.5 五折交叉验证 ====================
# 用 5 折交叉验证评估模型泛化能力的稳定性和真实水平（均值 + 标准差）
print("\n正在执行 5 折交叉验证...")
cv_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 计算 5 折的 R²（用负 MSE 取负号还原为正 MSE）
neg_mse_scores = cross_val_score(cv_model, X, y, cv=kfold, scoring='neg_mean_squared_error')
r2_scores = cross_val_score(cv_model, X, y, cv=kfold, scoring='r2')
mse_scores = -neg_mse_scores
rmse_scores = np.sqrt(mse_scores)

print(f"✅ 交叉验证完成")
print(f"   R²  均值: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
print(f"   RMSE 均值: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f} K")
print(f"   各折 R²: {[f'{s:.3f}' for s in r2_scores]}")

# ==================== 3.6 GridSearchCV 超参数调优 ====================
# 用 5 折交叉验证在给定参数网格中搜索最佳超参数，评分指标为 R²
print("\n正在执行 GridSearchCV 超参数调优（5 折交叉验证，评分 R²）...")

param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 1.0],
}

grid_model = XGBRegressor(random_state=42, n_jobs=-1)
grid = GridSearchCV(
    estimator=grid_model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=-1,
    verbose=0,
)
grid.fit(X, y)

print(f"✅ GridSearchCV 调优完成")
print(f"   最佳参数组合: {grid.best_params_}")
print(f"   最佳交叉验证 R²: {grid.best_score_:.3f}")

# ==================== 4. 训练模型 ====================
print("正在用最佳参数重新训练 XGBoost 模型...")

# 1. 初始化：用调优得到的最佳参数，early_stopping_rounds 写在构造函数里
model = XGBRegressor(
    n_estimators=grid.best_params_['n_estimators'],
    learning_rate=grid.best_params_['learning_rate'],
    max_depth=grid.best_params_['max_depth'],
    subsample=grid.best_params_['subsample'],
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

# 2. 训练：用独立验证集监控早停，测试集只留到最终评估
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # 监控验证集（不接触测试集）
    verbose=False                # 静默训练
)

print("✅ 模型训练完成")

# ==================== 5. 模型评估 ====================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 60)
print("📊 模型评估结果（测试集，最佳参数）")
print("=" * 60)
print(f"平均绝对误差 (MAE): {mae:.2f} K")
print(f"均方根误差 (RMSE): {rmse:.2f} K")
print(f"决定系数 (R²): {r2:.3f}")
print("=" * 60)

# ==================== 6. 特征重要性分析 ====================
print("\n📊 特征重要性分析（从高到低）:")
importances = model.feature_importances_
for name, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# ==================== 7. 可视化 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：实际值 vs 预测值散点图
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.5, s=10, c='steelblue')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('实际 Tg (K)', fontsize=12)
ax1.set_ylabel('预测 Tg (K)', fontsize=12)
ax1.set_title(f'预测效果 ($R^2$ = {r2:.3f})', fontsize=14)
ax1.grid(alpha=0.3)

# 图2：特征重要性柱状图
ax2 = axes[1]
sorted_idx = np.argsort(importances)
ax2.barh(np.array(X.columns)[sorted_idx], np.array(importances)[sorted_idx], color='steelblue')
ax2.set_xlabel('特征重要性', fontsize=12)
ax2.set_title('各结构特征的重要性', fontsize=14)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('Tg_Prediction_Results.png', dpi=300, bbox_inches='tight')
print("\n✅ 结果图已保存为 'Tg_Prediction_Results.png'")
plt.show()

print("\n" + "=" * 60)
print("🎉 项目运行完成！")
print("=" * 60)
