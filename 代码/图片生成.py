import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from _local_data import local_output

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.figsize'] = (10, 6)

# ------------------- 数据准备（全部来自您的Word文档）-------------------

# 表5 基准回归系数与标准误
base_vars = ['ROA', '营业利润率', '净利率', '成本费用率']
base_coef = [0.0302, 0.0611, 0.0549, -0.0341]
base_se = [0.0090, 0.0178, 0.0178, 0.0132]
base_sig = ['***', '***', '***', '**']

# 表6 滞后效应
lag_coef = [0.0158, 0.0253, 0.0211, -0.0094]
lag_se = [0.0069, 0.0173, 0.0182, 0.0138]

# 表9 TFP五种方法
tfp_methods = ['OP法', 'LP法', 'OLS法', 'FE法', 'GMM法']
tfp_coef = [-0.0057, -0.0023, -0.0003, -0.0030, -0.0092]
tfp_se = [0.0343, 0.0337, 0.0358, 0.0356, 0.0324]
tfp_p = [0.868, 0.945, 0.994, 0.932, 0.777]  # 均不显著

# 表10 异质性（只取表格中明确列出的分组）
het_labels = ['高文本证据', '低文本证据', '低数字基建', '高数字基建', '小规模企业', '大规模企业']
het_coef = [0.0275, 0.0253, 0.0408, 0.0211, 0.0289, 0.0265]
het_se = [0.0119, 0.0593, 0.0112, 0.0137, 0.0132, 0.0134]
het_sig = ['**', '', '***', '', '**', '*']  # 空字符串代表不显著

# 表11 稳健性
robust_labels = ['基准模型', '等权重法', '熵值法', '剔除疫情(2020-2021)', '滞后一期']
robust_coef = [0.0302, 0.0287, 0.0326, 0.0404, 0.0158]
robust_se = [0.0090, 0.0085, 0.0097, 0.0127, 0.0069]
robust_sig = ['***', '***', '***', '***', '**']

# 表7 机制检验（第一步）
mech_vars = ['成本费用率', '研发投入强度', '运营周转指数']
mech_coef = [-0.0341, 0.0026, -0.0427]
mech_se = [0.0132, 0.0025, 0.0849]
mech_sig = ['**', '', '']  # 仅成本费用率显著

# 表8 中介效应（第二步）
# 总效应 0.0302，直接效应 0.0134，间接效应 = 0.0168，占比 55.63%

# ------------------- 开始绘图（共8张）-------------------

# ---------- Figure 1: 基准效应 ----------
fig, ax = plt.subplots()
x = np.arange(len(base_vars))
bars = ax.bar(x, base_coef, yerr=[i*1.96 for i in base_se], capsize=5, color='steelblue', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(base_vars)
ax.set_ylabel('回归系数值')
ax.set_title('图1 基准回归：数据要素嵌入度对经营绩效的影响')
for i, (v, s) in enumerate(zip(base_coef, base_sig)):
    ax.text(i, v + 0.002 if v>0 else v - 0.008, s, ha='center', fontsize=12, color='red')
plt.tight_layout()
plt.savefig(local_output("Figure_1_基准效应.png"), dpi=300)
plt.show()

# ---------- Figure 2: 当期 vs 滞后效应 ----------
fig, ax = plt.subplots()
x = np.arange(len(base_vars))
width = 0.35
bars1 = ax.bar(x - width/2, base_coef, width, label='当期系数', yerr=[i*1.96 for i in base_se], capsize=3, color='royalblue')
bars2 = ax.bar(x + width/2, lag_coef, width, label='滞后一期系数', yerr=[i*1.96 for i in lag_se], capsize=3, color='tomato')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(base_vars)
ax.set_ylabel('回归系数值')
ax.set_title('图2 当期与滞后一期效应对比')
ax.legend()
plt.tight_layout()
plt.savefig(local_output("Figure_2_滞后效应对比.png"), dpi=300)
plt.show()

# ---------- Figure 3: 中介机制（系数变化瀑布图）----------
fig, ax = plt.subplots()
steps = ['总效应\n(c)', '间接效应\n(a*b)', '直接效应\n(c\')']
values = [0.0302, 0.0168, 0.0134]  # 间接效应 = 0.0302 - 0.0134
colors = ['steelblue', 'orange', 'steelblue']
bars = ax.bar(steps, values, color=colors, alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('ROA 回归系数')
ax.set_title('图3 成本效率的中介效应分解')
ax.text(0, 0.032, '***', ha='center', fontsize=12, color='red')
ax.text(1, 0.019, '占比 55.63%', ha='center', fontsize=10, color='darkred')
ax.text(2, 0.015, '**', ha='center', fontsize=12, color='red')
plt.tight_layout()
plt.savefig(local_output("Figure_3_中介效应分解.png"), dpi=300)
plt.show()

# ---------- Figure 4: TFP 生产率转化检验 ----------
fig, ax = plt.subplots()
bars = ax.bar(tfp_methods, tfp_coef, yerr=[i*1.96 for i in tfp_se], capsize=5, color='lightcoral', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_ylabel('回归系数值')
ax.set_title('图4 全要素生产率(TFP)转化检验（均不显著）')
for i, (v, p) in enumerate(zip(tfp_coef, tfp_p)):
    ax.text(i, v - 0.003 if v<0 else v + 0.002, f'p={p:.2f}', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(local_output("Figure_4_TFP转化检验.png"), dpi=300)
plt.show()

# ---------- Figure 5: 异质性分析 ----------
fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(len(het_labels))
bars = ax.bar(x, het_coef, yerr=[i*1.96 for i in het_se], capsize=5, color='mediumseagreen', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(het_labels, rotation=15)
ax.set_ylabel('ROA 回归系数')
ax.set_title('图5 异质性分析：赋能效应的边界条件')
for i, (v, s) in enumerate(zip(het_coef, het_sig)):
    y_pos = v + 0.003 if v>0 else v - 0.008
    ax.text(i, y_pos, s, ha='center', fontsize=12, color='red')
plt.tight_layout()
plt.savefig(local_output("Figure_5_异质性分析.png"), dpi=300)
plt.show()

# ---------- Figure 6: 稳健性检验 ----------
fig, ax = plt.subplots(figsize=(12,6))
x = np.arange(len(robust_labels))
bars = ax.bar(x, robust_coef, yerr=[i*1.96 for i in robust_se], capsize=5, color='darkorange', alpha=0.8)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(robust_labels, rotation=15)
ax.set_ylabel('ROA 回归系数')
ax.set_title('图6 稳健性检验：替换指标与剔除特殊年份')
for i, (v, s) in enumerate(zip(robust_coef, robust_sig)):
    ax.text(i, v + 0.002 if v>0 else v - 0.008, s, ha='center', fontsize=12, color='red')
plt.tight_layout()
plt.savefig(local_output("Figure_6_稳健性检验.png"), dpi=300)
plt.show()

# ---------- Figure 7: 机制检验（第一步：对三个机制变量的影响）----------
fig, ax = plt.subplots()
x = np.arange(len(mech_vars))
bars = ax.bar(x, mech_coef, yerr=[i*1.96 for i in mech_se], capsize=5, color='purple', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(mech_vars)
ax.set_ylabel('回归系数值')
ax.set_title('图7 机制检验第一步：DataEmbed对机制变量的影响')
for i, (v, s) in enumerate(zip(mech_coef, mech_sig)):
    y_pos = v + 0.002 if v>0 else v - 0.008
    ax.text(i, y_pos, s, ha='center', fontsize=12, color='red')
plt.tight_layout()
plt.savefig(local_output("Figure_7_机制第一步.png"), dpi=300)
plt.show()

# ---------- Figure 8: 中介效应占比饼图/条形图 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 左图：系数变化
steps = ['总效应', '间接效应', '直接效应']
vals = [0.0302, 0.0168, 0.0134]
colors = ['#2c3e50', '#e67e22', '#3498db']
ax1.bar(steps, vals, color=colors, alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='--')
ax1.set_ylabel('ROA 系数')
ax1.set_title('系数分解')
# 右图：占比饼图
labels = ['间接效应\n(成本效率)', '直接效应']
sizes = [55.63, 44.37]
explode = (0.1, 0)
ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#e67e22', '#3498db'])
ax2.set_title('中介效应占比')
fig.suptitle('图8 成本效率中介效应综合展示', fontsize=14)
plt.tight_layout()
plt.savefig(local_output("Figure_8_中介综合展示.png"), dpi=300)
plt.show()

print("所有图片已生成完毕！请查看当前目录下的 Figure_1.png ~ Figure_8.png")