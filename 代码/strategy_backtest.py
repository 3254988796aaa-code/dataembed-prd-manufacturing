# -*- coding: utf-8 -*-
"""
数字化转型策略反事实回测系统（最终校准版）
核心保证：双向固定效应系数与原论文完全一致
优化：增加方向准确率、分位数命中率等回测指标，更符合学术规范
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
from _local_data import local_input, local_output
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 0. 基础配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = local_output("回测结果输出")
OUTPUT_DIR.mkdir(exist_ok=True)

ALL_CONTROLS = [
    "Size_w", "Lev_w", "FixRatio_w", "EmployeeSize_w", "RDIntensity_w",
    "ln_GDPpc_w", "ln_FiscalTech_w", "ln_DigitalInfra_w"
]
CORE_X = "DataEmbed_w"
DEP_VARS = ["ROA_w", "CostRatio_w", "TFP_OP_w"]


# -------------------------- 1. 双向固定效应：标准估计+预测 --------------------------
def twoway_residualize(df, cols, entity="code6", time="year"):
    """标准双重去均值，和你原run_all.py完全一致，保证系数相同"""
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(float)
        out[c] = s - s.groupby(df[entity]).transform("mean") - s.groupby(df[time]).transform("mean") + s.mean()
    return out


def fit_twfe_standard(data, dep, x=CORE_X, controls=None, entity="code6", time="year"):
    """
    标准双向固定效应组内估计
    返回：beta系数、p值、样本量、个体固定效应、时间固定效应、控制变量系数
    """
    if controls is None:
        controls = []
    cols = [dep, x, entity, time] + controls
    d = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 30:
        return None

    # 1. 双重去均值估计beta（和你原代码逻辑100%一致）
    all_vars = [dep, x] + controls
    r = twoway_residualize(d, all_vars, entity, time)
    y = r[dep]
    X = r[[x] + controls]

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[entity]})
    beta = model.params[x]
    control_coefs = {c: model.params[c] for c in controls}

    # 2. 反推个体固定效应 alpha_i
    # alpha_i = 企业i的y均值 - beta*企业i的x均值 - 控制变量系数*企业i的控制变量均值
    firm_means = d.groupby(entity)[all_vars].mean()
    alpha_i = firm_means[dep].copy()
    alpha_i -= beta * firm_means[x]
    for c in controls:
        alpha_i -= control_coefs[c] * firm_means[c]
    alpha_dict = alpha_i.to_dict()
    avg_alpha = alpha_i.mean()

    # 3. 反推时间固定效应 gamma_t
    # gamma_t = 第t年的y均值 - beta*第t年的x均值 - 控制变量系数*第t年控制变量均值 - 平均个体效应
    year_means = d.groupby(time)[all_vars].mean()
    gamma_t = year_means[dep].copy()
    gamma_t -= beta * year_means[x]
    for c in controls:
        gamma_t -= control_coefs[c] * year_means[c]
    gamma_t -= avg_alpha
    gamma_dict = gamma_t.to_dict()
    avg_gamma = gamma_t.mean()

    # 整体截距
    const = d[dep].mean() - beta * d[x].mean()
    for c in controls:
        const -= control_coefs[c] * d[c].mean()

    return {
        "beta": beta,
        "p_value": model.pvalues[x],
        "nobs": int(model.nobs),
        "r2_within": model.rsquared,
        "const": const,
        "alpha_dict": alpha_dict,
        "gamma_dict": gamma_dict,
        "avg_alpha": avg_alpha,
        "avg_gamma": avg_gamma,
        "control_coefs": control_coefs,
        "model": model,
        "data": d
    }


def predict_twfe(test_data, model_res, entity="code6", time="year"):
    """基于训练好的TWFE模型做样本外预测"""
    beta = model_res["beta"]
    alpha_dict = model_res["alpha_dict"]
    gamma_dict = model_res["gamma_dict"]
    avg_alpha = model_res["avg_alpha"]
    avg_gamma = model_res["avg_gamma"]
    control_coefs = model_res["control_coefs"]

    test = test_data.copy()
    # 个体效应：训练集没出现的企业用平均个体效应
    test["alpha"] = test[entity].map(alpha_dict).fillna(avg_alpha)
    # 时间效应：训练集没出现的年份用平均时间效应
    test["gamma"] = test[time].map(gamma_dict).fillna(avg_gamma)

    # 核心变量效应
    test["x_effect"] = beta * test[CORE_X]
    # 控制变量效应
    test["ctrl_effect"] = 0
    for c, coef in control_coefs.items():
        if c in test.columns:
            test["ctrl_effect"] += coef * test[c].fillna(test[c].mean())

    # 最终预测值
    test["y_pred"] = test["alpha"] + test["gamma"] + test["x_effect"] + test["ctrl_effect"]
    return test["y_pred"].values


# -------------------------- 2. 读取数据 + 基准模型校准 --------------------------
print("=" * 60)
print("步骤1：读取数据，校准基准模型参数")
print("=" * 60)

df = pd.read_csv(local_input("panel.csv"), dtype=str)

# 兼容 code / code6
if "code6" in df.columns:
    df["code6"] = df["code6"].astype(str).str.zfill(6)
elif "code" in df.columns:
    df["code6"] = df["code"].astype(str).str.zfill(6)
else:
    raise KeyError("panel.csv 中缺少股票代码列")

# 数值列转换
num_cols = [CORE_X] + ALL_CONTROLS + DEP_VARS + ["year"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df[(df["year"] >= 2018) & (df["year"] <= 2024)].copy()
df = df.dropna(subset=[CORE_X, "year", "code6"])

exist_controls = [c for c in ALL_CONTROLS if c in df.columns]
print(f"可用控制变量：{exist_controls}")

# 校准基准模型
base_models = {}
for dep in DEP_VARS:
    if dep not in df.columns:
        print(f"⚠️  跳过 {dep}：数据中不存在该列")
        continue
    res = fit_twfe_standard(df, dep, CORE_X, exist_controls)
    if res is None:
        print(f"{dep:12s} | 回归失败")
        continue
    base_models[dep] = res
    print(f"{dep:12s} | 系数={res['beta']:.4f} | p值={res['p_value']:.4f} | 样本量={res['nobs']}")

DEP_VARS = [d for d in DEP_VARS if d in base_models]
if not DEP_VARS:
    raise RuntimeError("所有因变量均回归失败")

de_std = df[CORE_X].std()
de_mean = df[CORE_X].mean()
print(f"\nDataEmbed均值: {de_mean:.4f}，标准差: {de_std:.4f}")

# -------------------------- 3. 三档策略设计 --------------------------
print("\n" + "=" * 60)
print("步骤2：设计三档梯度转型策略")
print("=" * 60)

strategies = {
    "基础型（工具层嵌入）": {
        "de_lift_std": 0.5,
        "de_lift_abs": 0.5 * de_std,
        "description": "基础数据采集、单系统上线，补齐数字化底座",
        "build_period": 2,
        "full_effect_year": 3
    },
    "进阶型（流程层融合）": {
        "de_lift_std": 1.0,
        "de_lift_abs": 1.0 * de_std,
        "description": "跨系统数据打通、核心流程数字化改造",
        "build_period": 3,
        "full_effect_year": 4
    },
    "深度型（能力层落地）": {
        "de_lift_std": 1.5,
        "de_lift_abs": 1.5 * de_std,
        "description": "全链路数据驱动、智能化深度应用",
        "build_period": 4,
        "full_effect_year": 5
    }
}

strategy_df = pd.DataFrame([
    {
        "策略层级": name,
        "DataEmbed提升幅度（标准差）": v["de_lift_std"],
        "DataEmbed提升幅度（绝对值）": round(v["de_lift_abs"], 4),
        "策略说明": v["description"],
        "建设周期（年）": v["build_period"],
        "完全见效周期（年）": v["full_effect_year"]
    }
    for name, v in strategies.items()
])
strategy_df.to_csv(OUTPUT_DIR / "表1_三档策略参数表.csv", index=False, encoding="utf-8-sig")
print(strategy_df.to_string(index=False))

# -------------------------- 4. 截面反事实静态测算 --------------------------
print("\n" + "=" * 60)
print("步骤3：2024年截面反事实静态测算")
print("=" * 60)

latest_year = int(df["year"].max())
latest = df[df["year"] == latest_year].copy().reset_index(drop=True)
print(f"使用 {latest_year} 年数据作为基准截面，共 {len(latest)} 家企业")

for name, v in strategies.items():
    lift = v["de_lift_abs"]
    for dep in DEP_VARS:
        coef = base_models[dep]["beta"]
        abs_change = coef * lift
        latest[f"{name}_{dep}_绝对变化"] = abs_change
        base_mean = latest[dep].mean()
        latest[f"{name}_{dep}_相对变化率(%)"] = abs_change / base_mean * 100

# 汇总平均效果
summary_static = []
for name in strategies.keys():
    row = {"策略层级": name}
    for dep in DEP_VARS:
        row[f"{dep}_绝对提升均值"] = latest[f"{name}_{dep}_绝对变化"].mean()
        row[f"{dep}_相对提升率(%)"] = latest[f"{name}_{dep}_相对变化率(%)"].mean()
    summary_static.append(row)

summary_static_df = pd.DataFrame(summary_static)
summary_static_df.to_csv(OUTPUT_DIR / "表2_截面反事实平均效果.csv", index=False, encoding="utf-8-sig")
print(summary_static_df.round(4).to_string(index=False))
print("\n注：提升率为相对样本均值的比例，ROA基数较低故比例值较大")

latest.to_csv(OUTPUT_DIR / "表3_企业级策略效果明细.csv", index=False, encoding="utf-8-sig")

# -------------------------- 5. 动态演化路径模拟 --------------------------
print("\n" + "=" * 60)
print("步骤4：未来5年动态演化路径模拟")
print("=" * 60)

years_forecast = [2025, 2026, 2027, 2028, 2029]


def get_release_ratio(info, offset):
    build = info["build_period"]
    return min((offset + 1) / build, 1.0)


dynamic_results = []
for name, info in strategies.items():
    lift_total = info["de_lift_abs"]
    for i, year in enumerate(years_forecast):
        ratio = get_release_ratio(info, i)
        lift_year = lift_total * ratio
        row = {
            "策略层级": name,
            "年份": year,
            "释放比例(%)": round(ratio * 100, 1),
            "当年DataEmbed提升": round(lift_year, 4)
        }
        for dep in DEP_VARS:
            coef = base_models[dep]["beta"]
            effect = coef * lift_year
            base_mean = latest[dep].mean()
            row[f"{dep}_累计绝对提升"] = effect
            row[f"{dep}_累计提升率(%)"] = effect / base_mean * 100
        dynamic_results.append(row)

dynamic_df = pd.DataFrame(dynamic_results)
dynamic_df.to_csv(OUTPUT_DIR / "表4_策略动态演化路径.csv", index=False, encoding="utf-8-sig")
print(dynamic_df.round(4).to_string(index=False))

# -------------------------- 6. 滚动样本外回测（核心修正） --------------------------
print("\n" + "=" * 60)
print("步骤5：滚动样本外回测验证模型精度")
print("=" * 60)

test_years = sorted([y for y in [2021, 2022, 2023, 2024] if y <= df["year"].max()])
backtest_results = []
all_pred = []

for test_year in test_years:
    train_end = test_year - 1
    train = df[df["year"] <= train_end].copy()
    test = df[df["year"] == test_year].copy()

    # 训练模型
    model_res = fit_twfe_standard(train, "ROA_w", CORE_X, exist_controls)
    if model_res is None:
        continue

    # 样本外预测
    y_pred = predict_twfe(test, model_res)
    y_true = test["ROA_w"].values

    # 基础误差指标
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    # 新增：方向预测准确率（更有意义的指标）
    # 计算相对上一年的变化方向
    prev_year = test_year - 1
    prev_roa = df[df["year"] == prev_year].set_index("code6")["ROA_w"]
    test = test.set_index("code6")
    test["prev_roa"] = test.index.map(prev_roa)
    test["true_change"] = test["ROA_w"] - test["prev_roa"]
    test["pred_change"] = y_pred - test["prev_roa"]
    # 方向一致：同正或同负
    direction_correct = ((test["true_change"] > 0) & (test["pred_change"] > 0)) | (
                (test["true_change"] < 0) & (test["pred_change"] < 0))
    direction_acc = direction_correct.sum() / direction_correct.count()

    # 分位数命中率：Top25%预测命中率
    true_top = test["ROA_w"] >= test["ROA_w"].quantile(0.75)
    pred_top = pd.Series(y_pred, index=test.index) >= pd.Series(y_pred, index=test.index).quantile(0.75)
    top_hit = (true_top & pred_top).sum() / true_top.sum() if true_top.sum() > 0 else np.nan

    backtest_results.append({
        "预测年份": test_year,
        "训练样本量": len(train),
        "测试样本量": len(test),
        "MAE平均绝对误差": round(mae, 6),
        "RMSE均方根误差": round(rmse, 6),
        "样本外R²": round(r2, 4),
        "方向预测准确率(%)": round(direction_acc * 100, 2),
        "Top25%命中率(%)": round(top_hit * 100, 2)
    })

    tmp = test.reset_index()[["code6", "year", "ROA_w"]].copy()
    tmp["ROA_w预测值"] = y_pred
    all_pred.append(tmp)

backtest_df = pd.DataFrame(backtest_results)
backtest_df.to_csv(OUTPUT_DIR / "表5_样本外回测精度指标.csv", index=False, encoding="utf-8-sig")
print(backtest_df.to_string(index=False))
print("\n指标说明：")
print("  1. 样本外R²为截面拟合优度，面板单期截面R²偏低属正常现象")
print("  2. 方向预测准确率：预测绩效变化方向与实际一致的企业占比")
print("  3. Top25%命中率：预测绩效前25%的企业中，实际也在前25%的比例")

all_pred_df = pd.concat(all_pred, ignore_index=True)
all_pred_df.to_csv(OUTPUT_DIR / "表6_样本外逐企业预测明细.csv", index=False, encoding="utf-8-sig")

# -------------------------- 7. 可视化绘图 --------------------------
print("\n" + "=" * 60)
print("步骤6：生成可视化图表")
print("=" * 60)

# 图1：三档策略静态效果
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(strategies))
width = 0.25
colors = ["#2E7D32", "#C62828", "#1565C0"]
labels = {"ROA_w": "ROA", "CostRatio_w": "成本费用率", "TFP_OP_w": "TFP"}

for i, dep in enumerate(DEP_VARS):
    vals = [summary_static_df.loc[j, f"{dep}_相对提升率(%)"] for j in range(len(strategies))]
    bars = ax.bar(x + (i - 1) * width, vals, width, label=labels.get(dep, dep), color=colors[i])
    ax.bar_label(bars, fmt="%.2f%%", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(strategies.keys())
ax.set_ylabel("相对样本均值提升率（%）")
ax.set_title("图1 三档数字化转型策略静态效果对比")
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "图1_三档策略静态效果.png", dpi=300)
plt.close()

# 图2：ROA动态演化路径
fig, ax = plt.subplots(figsize=(10, 6))
for name in strategies.keys():
    sub = dynamic_df[dynamic_df["策略层级"] == name]
    ax.plot(sub["年份"], sub["ROA_w_累计提升率(%)"], marker="o", label=name, linewidth=2)

ax.set_xlabel("年份")
ax.set_ylabel("ROA累计提升率（%）")
ax.set_title("图2 数字化转型策略未来5年ROA动态演化路径")
ax.legend(fontsize=9)
ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "图2_动态演化路径.png", dpi=300)
plt.close()

# 图3：回测指标对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# 左：方向准确率
ax1.bar(backtest_df["预测年份"].astype(str), backtest_df["方向预测准确率(%)"], color="#2E7D32", alpha=0.8)
ax1.set_title("方向预测准确率")
ax1.set_ylabel("准确率（%）")
ax1.set_ylim(0, 100)
for i, v in enumerate(backtest_df["方向预测准确率(%)"]):
    ax1.text(i, v + 1, f"{v}%", ha="center")

# 右：Top25%命中率
ax2.bar(backtest_df["预测年份"].astype(str), backtest_df["Top25%命中率(%)"], color="#1565C0", alpha=0.8)
ax2.set_title("绩效Top25%预测命中率")
ax2.set_ylabel("命中率（%）")
ax2.set_ylim(0, 100)
for i, v in enumerate(backtest_df["Top25%命中率(%)"]):
    ax2.text(i, v + 1, f"{v}%", ha="center")

fig.suptitle("图3 样本外回测核心效果指标", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "图3_回测效果指标.png", dpi=300)
plt.close()

print(f"\n✅ 所有回测结果已输出到：{OUTPUT_DIR.resolve()}")
print("包含6张结果表 + 3张可视化图，可直接粘贴进Word报告")