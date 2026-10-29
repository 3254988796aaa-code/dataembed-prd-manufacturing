# -*- coding: utf-8 -*-
"""
运行方式：
    python 代码/run_all.py
    python scripts/run_final_pipeline.py --input_dir 输入数据 --output_dir local_outputs

说明：
1. 本脚本不写死绝对路径，可在任意解压目录运行。
2. 输出包括升级主面板、阶段诊断、案例卡片、回归结果、稳健性检验、图表与结果摘要。
3. 方法遵循“少量模型 + 强解释 + 可落地诊断”的竞赛作品逻辑。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from _local_data import INPUT_DIR, OUTPUT_DIR
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# 一、通用工具
# ---------------------------------------------------------------------

def project_root() -> Path:
    """自动识别项目根目录。

    公开版脚本位于代码目录，返回其上一级项目根目录。
    """
    return Path(__file__).resolve().parents[1]


def read_csv_smart(path: Path) -> pd.DataFrame:
    """兼容 utf-8-sig / utf-8 / gb18030 / gbk 的 CSV 读取。"""
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as exc:  # pragma: no cover
            last_err = exc
    raise RuntimeError(f"无法读取 CSV：{path}\n最后错误：{last_err}")


def to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def ensure_code6(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(".0", "")
    s = re.sub(r"\D", "", s)
    return s.zfill(6) if s else ""


def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\-－—_·•,，。、“”\"'\[\]【】()（）/\\：:;；]", "", s)
    return s.lower()


def sig_star(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def fmt_num(x, nd=4) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


def fmt_coef(coef, p=None, nd=4) -> str:
    if pd.isna(coef):
        return ""
    return f"{coef:.{nd}f}{sig_star(p)}"


# ---------------------------------------------------------------------
# 二、外部示范名单匹配
# ---------------------------------------------------------------------

def build_alias_table(mapping_df: pd.DataFrame) -> pd.DataFrame:
    """由上市公司简称、全称、子公司、品牌、曾用名构造别名库。"""
    rows = []
    mapping_df = mapping_df.copy()
    mapping_df["code6"] = mapping_df["股票代码"].apply(ensure_code6)
    for _, r in mapping_df.iterrows():
        code = r.get("code6", "")
        short = str(r.get("上市公司简称", "")).strip()
        full = str(r.get("上市公司全称", "")).strip()
        aliases = []
        for v in [short, full]:
            if v and v.lower() != "nan":
                aliases.append(v)
        text = str(r.get("主要子公司 / 品牌 / 曾用名", ""))
        text = text.replace("曾用名：", ";").replace("子公司：", ";").replace("品牌：", ";")
        text = text.replace("曾用名:", ";").replace("子公司:", ";").replace("品牌:", ";")
        for part in re.split(r"[;；、,，]+", text):
            part = part.strip()
            if part and part.lower() != "nan":
                aliases.append(part)
        for alias in set(aliases):
            norm = normalize_name(alias)
            if len(norm) >= 2 and norm != "nan":
                rows.append({
                    "code6": code,
                    "上市公司简称": short,
                    "上市公司全称": full,
                    "别名": alias,
                    "别名_norm": norm,
                })
    return pd.DataFrame(rows).drop_duplicates(["code6", "别名_norm"])


def match_external_lists(input_dir: Path, alias_df: pd.DataFrame) -> pd.DataFrame:
    """将政策/示范名单中的企业名称匹配到 99 家样本上市公司。"""
    specs = [
        ("industrial_internet_demo.csv", "IndustrialInternet_Demo", "工业互联网试点示范", "入选年度"),
        ("gd_smart_mfg_demo.csv", "SmartMfg_GD_Demo", "广东省智能制造示范", "入选年度"),
        ("national_smart_mfg_demo.csv", "SmartMfg_National_Demo", "国家级智能制造示范工厂", "入选年度"),
        ("little_giant_demo.csv", "LittleGiant", "国家级专精特新小巨人", "认定年度"),
    ]
    alias_records = alias_df.to_dict("records")
    matches: List[Dict] = []
    for fname, var, label, year_col in specs:
        path = input_dir / fname
        if not path.exists():
            continue
        ext = read_csv_smart(path)
        for _, er in ext.iterrows():
            ext_name = str(er.get("企业名称", "")).strip()
            norm_ext = normalize_name(ext_name)
            best, best_score = None, -1
            for ar in alias_records:
                alias_norm = ar["别名_norm"]
                if not alias_norm:
                    continue
                if len(alias_norm) <= 2 and alias_norm != norm_ext:
                    continue
                hit = (alias_norm in norm_ext) or (norm_ext in alias_norm)
                if hit:
                    score = min(len(alias_norm), len(norm_ext))
                    if score > best_score:
                        best_score = score
                        best = ar
            if best is not None:
                project_name = er.get("项目名称", er.get("项目 / 工厂名称", er.get("示范工厂名称", er.get("所属行业", ""))))
                matches.append({
                    "名单来源文件": fname,
                    "标签变量": var,
                    "标签名称": label,
                    "入选年度": er.get(year_col, np.nan),
                    "名单城市": er.get("城市", ""),
                    "名单企业名称": ext_name,
                    "名单项目/行业": project_name,
                    "code6": best["code6"],
                    "匹配上市公司简称": best["上市公司简称"],
                    "匹配上市公司全称": best["上市公司全称"],
                    "匹配别名": best["别名"],
                    "匹配分数": best_score,
                    "匹配说明": "名称包含/被包含匹配，短简称仅完全相等",
                })
    if not matches:
        return pd.DataFrame()
    return pd.DataFrame(matches).drop_duplicates()


def add_policy_flags(panel: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    labels = ["IndustrialInternet_Demo", "SmartMfg_GD_Demo", "SmartMfg_National_Demo", "LittleGiant"]
    flags = panel[["code6", "year"]].copy()
    for lab in labels:
        flags[lab] = 0
    first_year = {}
    for code in panel["code6"].unique():
        sub = matches[matches["code6"] == code] if len(matches) else pd.DataFrame()
        first_year[code] = pd.to_numeric(sub.get("入选年度", pd.Series(dtype=float)), errors="coerce").min() if len(sub) else np.nan
    for _, m in matches.iterrows():
        y = pd.to_numeric(m.get("入选年度", np.nan), errors="coerce")
        if pd.isna(y):
            y = 2017
        lab = m.get("标签变量", "")
        if lab in flags.columns:
            mask = (flags["code6"] == m["code6"]) & (flags["year"] >= int(y))
            flags.loc[mask, lab] = 1
    flags["PolicyDemo_Any"] = flags[labels].max(axis=1)
    flags["PolicyDemo_Count"] = flags[labels].sum(axis=1)
    flags["FirstDemoYear"] = flags["code6"].map(first_year)
    out = panel.merge(flags, on=["code6", "year"], how="left")
    for c in labels + ["PolicyDemo_Any", "PolicyDemo_Count"]:
        out[c] = out[c].fillna(0).astype(int)
    return out


# ---------------------------------------------------------------------
# 三、双向固定效应与稳健性检验
# ---------------------------------------------------------------------

DEFAULT_CONTROLS = [
    "Size_w", "Lev_w", "FixRatio_w", "EmployeeSize_w", "RDIntensity_w",
    "ln_GDPpc_w", "ln_FiscalTech_w", "ln_DigitalInfra_w"
]


def twoway_residualize(df: pd.DataFrame, cols: List[str], entity="code6", time="year") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(float)
        out[c] = s - s.groupby(df[entity]).transform("mean") - s.groupby(df[time]).transform("mean") + s.mean()
    return out


def fe_regression(data: pd.DataFrame, dep: str, x: str = "DataEmbed_w", controls: Optional[List[str]] = None, extra: Optional[List[str]] = None, model_name: str = "TWFE") -> Dict:
    if controls is None:
        controls = []
    if extra is None:
        extra = []
    cols = [dep, x] + controls + extra
    available = [c for c in cols if c in data.columns]
    if dep not in available or x not in available:
        return {"模型": model_name, "因变量": dep, "核心变量": x, "样本量": 0, "系数": np.nan, "标准误": np.nan, "p值": np.nan, "R2(去均值)": np.nan, "显著性": ""}
    d = data[available + ["code6", "year"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 30 or d[x].std() <= 1e-12:
        return {"模型": model_name, "因变量": dep, "核心变量": x, "样本量": len(d), "系数": np.nan, "标准误": np.nan, "p值": np.nan, "R2(去均值)": np.nan, "显著性": ""}
    r = twoway_residualize(d, available)
    y = r[dep]
    X_cols = [c for c in [x] + controls + extra if c in r.columns and r[c].std() > 1e-12]
    if x not in X_cols:
        return {"模型": model_name, "因变量": dep, "核心变量": x, "样本量": len(d), "系数": np.nan, "标准误": np.nan, "p值": np.nan, "R2(去均值)": np.nan, "显著性": ""}
    X = r[X_cols]
    try:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d["code6"]})
        coef = model.params.get(x, np.nan)
        se = model.bse.get(x, np.nan)
        p = model.pvalues.get(x, np.nan)
        r2 = model.rsquared
    except Exception:
        model = sm.OLS(y, X).fit()
        coef = model.params.get(x, np.nan)
        se = model.bse.get(x, np.nan)
        p = model.pvalues.get(x, np.nan)
        r2 = model.rsquared
    return {
        "模型": model_name,
        "因变量": dep,
        "核心变量": x,
        "控制变量数": len([c for c in controls + extra if c in X_cols]),
        "样本量": int(model.nobs),
        "系数": float(coef) if pd.notna(coef) else np.nan,
        "标准误": float(se) if pd.notna(se) else np.nan,
        "p值": float(p) if pd.notna(p) else np.nan,
        "R2(去均值)": float(r2) if pd.notna(r2) else np.nan,
        "显著性": sig_star(p),
        "系数(含显著性)": fmt_coef(coef, p),
    }


def run_regressions(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    main = panel[(panel["year"] >= 2018) & (panel["year"] <= 2024)].copy()
    core_deps = ["ROA_w", "OperatingMargin_w", "NetMargin_w", "CostRatio_w", "CostEfficiency_w", "TFP_OP_w", "TFP_LP_w", "TFP_OLS_w", "TFP_FE_w", "TFP_GMM_w"]
    core = pd.DataFrame([fe_regression(main, dep, "DataEmbed_w", DEFAULT_CONTROLS, model_name="核心TWFE") for dep in core_deps])
    lag = pd.DataFrame([fe_regression(main, dep, "L1_DataEmbed_w", DEFAULT_CONTROLS, model_name="一期滞后TWFE") for dep in ["ROA_w", "OperatingMargin_w", "NetMargin_w", "CostRatio_w", "TFP_OP_w"]])
    mechanism_cost = pd.DataFrame([
        fe_regression(main, "CostRatio_w", "DataEmbed_w", DEFAULT_CONTROLS, model_name="成本机制"),
        fe_regression(main, "CostEfficiency_w", "DataEmbed_w", DEFAULT_CONTROLS, model_name="成本机制"),
        fe_regression(main, "TurnoverIndex_w", "DataEmbed_w", DEFAULT_CONTROLS, model_name="营运机制"),
    ])
    mechanism_added = pd.DataFrame([fe_regression(main, dep, "DataEmbed_w", DEFAULT_CONTROLS, extra=["CostRatio_w"], model_name="加入成本费用率") for dep in ["ROA_w", "OperatingMargin_w", "NetMargin_w"]])
    base_roa = core.loc[core["因变量"] == "ROA_w", "系数"].iloc[0]
    after_roa = mechanism_added.loc[mechanism_added["因变量"] == "ROA_w", "系数"].iloc[0]
    contribution = (base_roa - after_roa) / base_roa if pd.notna(base_roa) and abs(base_roa) > 1e-12 else np.nan
    mechanism_summary = pd.DataFrame([{
        "机制口径": "以 ROA 为例：加入 CostRatio 后 DataEmbed 系数下降比例",
        "基准系数": base_roa,
        "加入成本费用率后系数": after_roa,
        "成本效率通道贡献率": contribution,
        "解释": "贡献率仅表示系数变化比例，用作机制线索，不作严格因果分解。",
    }])

    # 异质性：证据强度、规模、研发、数字基础设施。
    het_rows = []
    for g in ["HighEvidence", "HighSize", "HighRD", "HighDigitalInfra"]:
        if g not in main.columns:
            continue
        for val, label in [(0, "低组"), (1, "高组")]:
            sub = main[main[g] == val].copy()
            res = fe_regression(sub, "ROA_w", "DataEmbed_w", DEFAULT_CONTROLS, model_name="异质性TWFE")
            res.update({"分组变量": g, "组别": label})
            het_rows.append(res)
    heterogeneity = pd.DataFrame(het_rows)

    # 稳健性1：替代解释变量口径（等权、熵权）。
    robust_rows = []
    for x in ["DataEmbed_w", "DataEmbed_equal_w", "DataEmbed_entropy_w"]:
        for dep in ["ROA_w", "OperatingMargin_w", "NetMargin_w", "CostRatio_w", "TFP_OP_w"]:
            robust_rows.append(fe_regression(main, dep, x, DEFAULT_CONTROLS, model_name="替代指标稳健性"))
    robust_alt_index = pd.DataFrame(robust_rows)

    return {
        "core": core,
        "lag": lag,
        "mechanism_cost": mechanism_cost,
        "mechanism_added": mechanism_added,
        "mechanism_summary": mechanism_summary,
        "heterogeneity": heterogeneity,
        "robust_alt_index": robust_alt_index,
    }


def placebo_test(panel: pd.DataFrame, reps: int = 120, seed: int = 2026) -> pd.DataFrame:
    """按年份内随机置换 DataEmbed，用于安慰剂检验。"""
    rng = np.random.default_rng(seed)
    main = panel[(panel["year"] >= 2018) & (panel["year"] <= 2024)].copy()
    obs = fe_regression(main, "ROA_w", "DataEmbed_w", DEFAULT_CONTROLS, model_name="真实样本")
    rows = [{"序号": 0, "类型": "真实样本", "系数": obs["系数"], "p值": obs["p值"]}]
    for i in range(1, reps + 1):
        tmp = main.copy()
        shuffled = []
        for _, g in tmp.groupby("year"):
            vals = g["DataEmbed_w"].to_numpy().copy()
            rng.shuffle(vals)
            shuffled.extend(vals)
        tmp["DataEmbed_placebo"] = shuffled
        res = fe_regression(tmp, "ROA_w", "DataEmbed_placebo", DEFAULT_CONTROLS, model_name="安慰剂")
        rows.append({"序号": i, "类型": "年份内随机置换", "系数": res["系数"], "p值": res["p值"]})
    out = pd.DataFrame(rows)
    placebo = out[out["类型"] != "真实样本"]
    pseudo_p = (np.abs(placebo["系数"]) >= abs(obs["系数"])).mean() if len(placebo) else np.nan
    out["真实系数绝对值更极端的置换比例"] = pseudo_p
    return out


# ---------------------------------------------------------------------
# 四、阶段诊断、应用输出与规则库
# ---------------------------------------------------------------------

def add_application_scores(latest: pd.DataFrame) -> pd.DataFrame:
    latest = latest.copy()
    p_de = latest["DataEmbed_w"].rank(pct=True)
    p_cost = latest["CostRatio_w"].rank(pct=True)
    p_roa_low = 1 - latest["ROA_w"].rank(pct=True)
    p_tfp_low = 1 - latest["TFP_OP_w"].rank(pct=True)
    evidence_score = latest.get("evidence_level", pd.Series(0, index=latest.index)).fillna(0) / max(latest.get("evidence_level", pd.Series([1])).max(), 1)
    demo_score = latest.get("PolicyDemo_Count_Ever", pd.Series(0, index=latest.index)).fillna(0)
    if demo_score.max() > 0:
        demo_score = demo_score / demo_score.max()
    infra_score = latest.get("ln_DigitalInfra_w", pd.Series(0, index=latest.index)).rank(pct=True)
    latest["应用成熟度得分"] = (40 * p_de + 20 * evidence_score + 20 * demo_score + 20 * infra_score).round(2)
    latest["赋能转化风险得分"] = (35 * p_cost + 35 * p_roa_low + 30 * p_tfp_low).round(2)
    latest["优先级"] = pd.cut(latest["赋能转化风险得分"], bins=[-np.inf, 40, 65, np.inf], labels=["常规跟踪", "重点辅导", "优先干预"])
    return latest


def classify_stage(panel: pd.DataFrame, q_low: float = 0.25, q_mid: float = 0.50, q_tfp: float = 0.65) -> pd.DataFrame:
    latest = panel[panel["year"] == 2024].copy()
    any_by = panel.groupby("code6")["PolicyDemo_Any"].max()
    count_by = panel.groupby("code6")["PolicyDemo_Count"].max()
    latest["PolicyDemo_Ever"] = latest["code6"].map(any_by).fillna(0).astype(int)
    latest["PolicyDemo_Count_Ever"] = latest["code6"].map(count_by).fillna(0).astype(int)

    wide = panel[panel["year"].isin([2018, 2024])].pivot_table(index="code6", columns="year", values=["DataEmbed_w", "ROA_w", "CostRatio_w", "TFP_OP_w"], aggfunc="first")
    impr = pd.DataFrame(index=wide.index)
    for var in ["DataEmbed_w", "ROA_w", "CostRatio_w", "TFP_OP_w"]:
        try:
            impr[f"2018-2024变化_{var}"] = wide[(var, 2024)] - wide[(var, 2018)]
        except Exception:
            impr[f"2018-2024变化_{var}"] = np.nan
    latest = latest.merge(impr, left_on="code6", right_index=True, how="left")

    de_low = latest["DataEmbed_w"].quantile(q_low)
    de_mid = latest["DataEmbed_w"].quantile(q_mid)
    roa_mid = latest["ROA_w"].quantile(0.50)
    cost_mid = latest["CostRatio_w"].quantile(0.50)
    tfp_cut = latest["TFP_OP_w"].quantile(q_tfp)

    def _stage(r: pd.Series) -> str:
        de, roa, cost, tfp = r.get("DataEmbed_w", np.nan), r.get("ROA_w", np.nan), r.get("CostRatio_w", np.nan), r.get("TFP_OP_w", np.nan)
        droa, dcost, dtfp = r.get("2018-2024变化_ROA_w", np.nan), r.get("2018-2024变化_CostRatio_w", np.nan), r.get("2018-2024变化_TFP_OP_w", np.nan)
        if pd.isna(de):
            return "资料补全期"
        if de <= de_low:
            return "基础补课期"
        if de >= de_mid and ((tfp >= tfp_cut) or (pd.notna(dtfp) and dtfp > 0.15)) and ((roa >= roa_mid) or (pd.notna(droa) and droa > 0)):
            return "转化观察期"
        if de >= de_mid and ((cost <= cost_mid) or (pd.notna(dcost) and dcost < 0)):
            return "成本优化期"
        if de >= de_low and ((roa >= roa_mid) or (pd.notna(droa) and droa > 0)):
            return "经营改善期"
        return "系统嵌入期"

    advice_map = {
        "基础补课期": "优先补齐数据采集、系统连接和基础台账，避免一开始投入过重的智能化项目。",
        "系统嵌入期": "围绕 ERP/MES/PLM/供应链系统打通数据链路，将年报数字化表述转化为流程级应用。",
        "经营改善期": "将数据应用从管理可视化推进到订单响应、质量追溯和客户服务，持续跟踪 ROA 与利润率。",
        "成本优化期": "重点攻关库存、排产、能耗、采购与质量损失，形成可量化的成本效率改善闭环。",
        "转化观察期": "进入生产率转化观察窗口，建议把流程再造、组织学习和数据治理能力纳入中长期考核。",
        "资料补全期": "补充企业年报文本、财务和 TFP 数据后再做阶段判断。",
    }
    latest["阶段诊断2024"] = latest.apply(_stage, axis=1)
    latest["诊断建议"] = latest["阶段诊断2024"].map(advice_map)
    latest["数据嵌入排名百分位"] = latest["DataEmbed_w"].rank(pct=True)
    latest = add_application_scores(latest)
    cols = [
        "code6", "short_name", "firm_name", "city", "行业代码D", "DataEmbed_w", "DataEmbed_equal_w", "DataEmbed_entropy_w",
        "ROA_w", "OperatingMargin_w", "NetMargin_w", "CostRatio_w", "CostEfficiency_w", "TFP_OP_w", "TFP_LP_w",
        "RDIntensity_w", "evidence_level", "evidence_sentence", "PolicyDemo_Ever", "PolicyDemo_Count_Ever",
        "IndustrialInternet_Demo", "SmartMfg_GD_Demo", "SmartMfg_National_Demo", "LittleGiant",
        "2018-2024变化_DataEmbed_w", "2018-2024变化_ROA_w", "2018-2024变化_CostRatio_w", "2018-2024变化_TFP_OP_w",
        "数据嵌入排名百分位", "应用成熟度得分", "赋能转化风险得分", "优先级", "阶段诊断2024", "诊断建议"
    ]
    cols = [c for c in cols if c in latest.columns]
    return latest[cols].sort_values(["优先级", "阶段诊断2024", "DataEmbed_w"], ascending=[False, True, False])


def threshold_sensitivity(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q_low in [0.20, 0.25, 0.30]:
        for q_mid in [0.45, 0.50, 0.55]:
            for q_tfp in [0.60, 0.65, 0.70]:
                diag = classify_stage(panel, q_low, q_mid, q_tfp)
                dist = diag["阶段诊断2024"].value_counts().to_dict()
                row = {"低嵌入阈值分位": q_low, "中高嵌入阈值分位": q_mid, "TFP观察阈值分位": q_tfp}
                for stage in ["基础补课期", "系统嵌入期", "经营改善期", "成本优化期", "转化观察期"]:
                    row[stage] = dist.get(stage, 0)
                rows.append(row)
    return pd.DataFrame(rows)


def make_case_cards(diagnosis: pd.DataFrame) -> pd.DataFrame:
    picks = []
    demo = diagnosis[diagnosis["PolicyDemo_Ever"] == 1].sort_values(["PolicyDemo_Count_Ever", "DataEmbed_w"], ascending=False).head(8)
    risk = diagnosis.sort_values("赋能转化风险得分", ascending=False).head(6)
    high = diagnosis.sort_values("应用成熟度得分", ascending=False).head(6)
    for _, r in pd.concat([demo, risk, high]).drop_duplicates("code6").head(16).iterrows():
        strengths = []
        if r.get("PolicyDemo_Ever", 0) == 1:
            strengths.append("外部示范标签")
        if r.get("evidence_level", 0) >= 4:
            strengths.append("文本落地证据强")
        if r.get("DataEmbed_w", 0) >= diagnosis["DataEmbed_w"].quantile(0.75):
            strengths.append("数据嵌入度较高")
        if r.get("CostRatio_w", 999) <= diagnosis["CostRatio_w"].median():
            strengths.append("成本费用率相对较低")
        if not strengths:
            strengths.append("具备进一步诊断价值")
        picks.append({
            "股票代码": r.get("code6", ""),
            "企业简称": r.get("short_name", ""),
            "城市": r.get("city", ""),
            "阶段": r.get("阶段诊断2024", ""),
            "优先级": r.get("优先级", ""),
            "应用成熟度得分": r.get("应用成熟度得分", np.nan),
            "赋能转化风险得分": r.get("赋能转化风险得分", np.nan),
            "DataEmbed_w": r.get("DataEmbed_w", np.nan),
            "ROA_w": r.get("ROA_w", np.nan),
            "CostRatio_w": r.get("CostRatio_w", np.nan),
            "TFP_OP_w": r.get("TFP_OP_w", np.nan),
            "外部示范标签数": r.get("PolicyDemo_Count_Ever", 0),
            "主要优势/风险": "；".join(strengths),
            "建议动作": r.get("诊断建议", ""),
            "年报证据句": r.get("evidence_sentence", ""),
        })
    return pd.DataFrame(picks)


def make_decision_rules() -> pd.DataFrame:
    return pd.DataFrame([
        {"阶段": "基础补课期", "识别条件": "DataEmbed 位于低分位，系统连接与年报证据不足", "核心痛点": "数据采集和数据治理基础薄弱", "优先动作": "统一主数据、补全设备/订单/质量台账，建设轻量数据看板", "不建议动作": "直接上复杂AI模型或大规模平台建设", "评估指标": "数据完整率、系统接入数、基础指标更新频率"},
        {"阶段": "系统嵌入期", "识别条件": "DataEmbed 已越过低分位但经营改善尚不明显", "核心痛点": "系统上线但流程没有真正重构", "优先动作": "打通 ERP/MES/PLM/供应链系统，建立跨部门指标口径", "不建议动作": "只展示数字化宣传材料而不做流程穿透", "评估指标": "跨系统字段匹配率、流程在线化率、异常工单闭环率"},
        {"阶段": "经营改善期", "识别条件": "数据嵌入提升且 ROA/利润率相对改善", "核心痛点": "从管理可视化到业务动作仍需闭环", "优先动作": "围绕订单响应、客户服务、质量追溯输出月度行动清单", "不建议动作": "只停留在财务结果复盘", "评估指标": "ROA、营业利润率、客户交付周期"},
        {"阶段": "成本优化期", "识别条件": "DataEmbed 较高且成本费用率低于中位或下降", "核心痛点": "成本效率改善需要跨环节协同", "优先动作": "排产优化、库存周转、采购价格、质量损失四类专题联动", "不建议动作": "只做单点降本，忽视交付和质量", "评估指标": "成本费用率、库存周转、单位能耗、质量损失率"},
        {"阶段": "转化观察期", "识别条件": "DataEmbed、经营绩效和 TFP 表现均相对较好", "核心痛点": "需要把短期经营改善沉淀为生产率能力", "优先动作": "沉淀数据治理制度、算法复用资产和组织学习机制", "不建议动作": "过度追求短期财务显著性", "评估指标": "TFP 多口径趋势、算法复用率、流程再造收益"},
    ])


def make_user_checklist() -> pd.DataFrame:
    return pd.DataFrame([
        {"用户类型": "企业经营管理者", "使用入口": "企业诊断2024.csv / Excel仪表盘", "查看重点": "阶段、优先级、风险得分、建议动作", "推荐决策": "确定下一年度数据治理和降本增效项目池"},
        {"用户类型": "数字化部门", "使用入口": "升级主面板与指标字段", "查看重点": "DataEmbed、证据等级、系统/平台/落地关键词密度", "推荐决策": "检查年报披露与实际系统建设是否一致"},
        {"用户类型": "政府/园区", "使用入口": "政策示范匹配结果与风险优先级", "查看重点": "外部示范标签、城市、行业、赋能转化风险", "推荐决策": "筛选重点辅导企业和标杆培育对象"},
        {"用户类型": "参赛评审/复现者", "使用入口": "README 与 scripts/run_final_pipeline.py", "查看重点": "输入文件、输出文件、随机种子、回归表", "推荐决策": "一键复现结果并检查模型稳定性"},
    ])


# ---------------------------------------------------------------------
# 五、外部验证与可视化
# ---------------------------------------------------------------------

def external_validation(diagnosis: pd.DataFrame) -> pd.DataFrame:
    x1 = diagnosis.loc[diagnosis["PolicyDemo_Ever"] == 1, "DataEmbed_w"].dropna()
    x0 = diagnosis.loc[diagnosis["PolicyDemo_Ever"] == 0, "DataEmbed_w"].dropna()
    t = stats.ttest_ind(x1, x0, equal_var=False) if len(x1) and len(x0) else None
    mw = stats.mannwhitneyu(x1, x0, alternative="two-sided") if len(x1) and len(x0) else None
    pooled_sd = math.sqrt(((len(x1)-1)*x1.var() + (len(x0)-1)*x0.var()) / (len(x1)+len(x0)-2)) if len(x1)>1 and len(x0)>1 else np.nan
    cohen_d = (x1.mean() - x0.mean()) / pooled_sd if pooled_sd and not pd.isna(pooled_sd) and pooled_sd != 0 else np.nan
    rows = [
        {"检验项": "命中示范组样本量", "数值": len(x1), "说明": "2024年企业中至少命中一种外部示范标签"},
        {"检验项": "未命中组样本量", "数值": len(x0), "说明": "2024年企业中未命中外部示范标签"},
        {"检验项": "命中组DataEmbed均值", "数值": x1.mean(), "说明": "仅作方向性有效性参照"},
        {"检验项": "未命中组DataEmbed均值", "数值": x0.mean(), "说明": "仅作方向性有效性参照"},
        {"检验项": "均值差", "数值": x1.mean()-x0.mean(), "说明": "正值表示示范组嵌入度更高"},
        {"检验项": "Welch t统计量", "数值": getattr(t, "statistic", np.nan), "说明": "两组方差不等的均值差检验"},
        {"检验项": "Welch t检验p值", "数值": getattr(t, "pvalue", np.nan), "说明": "不显著时不作强验证"},
        {"检验项": "Mann-Whitney U统计量", "数值": getattr(mw, "statistic", np.nan), "说明": "非参数秩和检验"},
        {"检验项": "Mann-Whitney p值", "数值": getattr(mw, "pvalue", np.nan), "说明": "不要求正态分布"},
        {"检验项": "Cohen d", "数值": cohen_d, "说明": "效应量；小样本仅辅助解释"},
    ]
    return pd.DataFrame(rows)


def setup_chinese_font() -> None:
    candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei", "AR PL UMing CN"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.sans-serif"] = [c]
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_fig(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(panel: pd.DataFrame, diagnosis: pd.DataFrame, regs: Dict[str, pd.DataFrame], robust_alt: pd.DataFrame, placebo: pd.DataFrame, fig_dir: Path) -> None:
    setup_chinese_font()
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 图1：竞赛型作品总体架构
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    layers = [
        ("问题层", "数据要素投入多\n转化路径不清\n企业差异大"),
        ("数据层", "年报文本\n财务数据\n城市数字基础设施\n政策示范名单"),
        ("模型层", "DataEmbed 指标\n双向固定效应\n机制/稳健性检验"),
        ("诊断层", "五阶段识别\n风险优先级\n案例卡片"),
        ("决策层", "企业行动清单\n园区扶持筛选\n可复现仪表盘"),
    ]
    xs = np.linspace(0.08, 0.92, len(layers))
    for i, (title, body) in enumerate(layers):
        ax.text(xs[i], 0.78, title, ha="center", va="center", fontsize=16, weight="bold", bbox=dict(boxstyle="round,pad=0.45", fc="#DDEBFF", ec="#2A5CAA", lw=1.5))
        ax.text(xs[i], 0.46, body, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.55", fc="#F7FAFF", ec="#7BA1D2", lw=1.2))
        if i < len(layers) - 1:
            ax.annotate("", xy=(xs[i+1]-0.08, 0.58), xytext=(xs[i]+0.08, 0.58), arrowprops=dict(arrowstyle="->", lw=2, color="#2A5CAA"))
    ax.text(0.5, 0.12, "作品定位：从论文式显著性检验升级为“数据要素诊断 + 阶段赋能 + 决策支持”的可落地系统", ha="center", fontsize=14, color="#A85D00", weight="bold")
    save_fig(fig, fig_dir / "图1_竞赛作品总体架构图.png")

    # 图2：数据处理与交付流程
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.axis("off")
    steps = ["原始数据\n读取", "股票代码/名称\n标准化", "企业别名\n匹配", "政策示范\n标签构造", "实证与稳健性\n检验", "阶段诊断\n与应用输出"]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, s in enumerate(steps):
        ax.text(xs[i], 0.55, s, ha="center", va="center", fontsize=13, bbox=dict(boxstyle="round,pad=0.48", fc="#FFF3D6", ec="#D99B00", lw=1.5))
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i+1]-0.07, 0.55), xytext=(xs[i]+0.07, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color="#D99B00"))
    ax.text(0.5, 0.18, "输出：主面板、诊断表、案例卡片、回归表、稳健性表、规则库、用户清单、图表、最终文章", ha="center", fontsize=13)
    save_fig(fig, fig_dir / "图2_数据处理与交付流程图.png")

    # 图3：DataEmbed 指标体系
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    center = (0.5, 0.5)
    ax.text(*center, "DataEmbed\n数据要素嵌入度", ha="center", va="center", fontsize=16, weight="bold", bbox=dict(boxstyle="circle,pad=0.45", fc="#EAF7EA", ec="#2E7D32", lw=2))
    items = ["数字化\n关键词密度", "智能制造\n关键词密度", "数据治理\n关键词密度", "系统平台\n关键词密度", "落地实施\n关键词密度", "年报证据\n等级"]
    angles = np.linspace(0, 2*np.pi, len(items), endpoint=False)
    for a, item in zip(angles, items):
        x = center[0] + 0.34*np.cos(a)
        y = center[1] + 0.34*np.sin(a)
        ax.text(x, y, item, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", fc="#F9FFF9", ec="#66A96B", lw=1.2))
        ax.annotate("", xy=(center[0]+0.12*np.cos(a), center[1]+0.12*np.sin(a)), xytext=(x-0.08*np.cos(a), y-0.08*np.sin(a)), arrowprops=dict(arrowstyle="->", lw=1.5, color="#66A96B"))
    ax.text(0.5, 0.08, "多维文本指标 + 标准化处理 + 替代口径稳健性，降低单一词频偏误", ha="center", fontsize=13, color="#2E7D32")
    save_fig(fig, fig_dir / "图3_DataEmbed指标体系图.png")

    # 图4：阶段性赋能机制
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.axis("off")
    stages = [("基础补课", "数据采集\n主数据治理"), ("系统嵌入", "ERP/MES/PLM\n平台连接"), ("经营改善", "ROA\n利润率"), ("成本优化", "库存/排产\n采购/质量"), ("转化观察", "TFP\n组织学习")]
    xs = np.linspace(0.10, 0.90, len(stages))
    for i, (t, b) in enumerate(stages):
        ax.text(xs[i], 0.66, t, ha="center", va="center", fontsize=14, weight="bold", bbox=dict(boxstyle="round,pad=0.42", fc="#E8F1FF", ec="#3F6DB3", lw=1.5))
        ax.text(xs[i], 0.37, b, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.50", fc="#FFFFFF", ec="#A6BCE0", lw=1.2))
        if i < len(stages)-1:
            ax.annotate("", xy=(xs[i+1]-0.10, 0.52), xytext=(xs[i]+0.10, 0.52), arrowprops=dict(arrowstyle="->", lw=2, color="#3F6DB3"))
    ax.text(0.5, 0.12, "核心判断：经营绩效和成本效率更快显现，TFP 转化需要更长周期与组织配套", ha="center", fontsize=13, color="#B35400", weight="bold")
    save_fig(fig, fig_dir / "图4_阶段性赋能机制图.png")

    # 图5：核心系数图
    core = regs["core"].copy()
    name_map = {"ROA_w": "ROA", "OperatingMargin_w": "营业利润率", "NetMargin_w": "净利率", "CostRatio_w": "成本费用率", "CostEfficiency_w": "成本效率", "TFP_OP_w": "TFP_OP", "TFP_LP_w": "TFP_LP", "TFP_OLS_w": "TFP_OLS", "TFP_FE_w": "TFP_FE", "TFP_GMM_w": "TFP_GMM"}
    plot = core[core["因变量"].isin(name_map)].copy()
    plot["变量"] = plot["因变量"].map(name_map)
    fig, ax = plt.subplots(figsize=(9.8, 6.5))
    y = np.arange(len(plot))
    ax.errorbar(plot["系数"], y, xerr=1.96*plot["标准误"], fmt="o", capsize=4, color="#255C99", ecolor="#777777")
    ax.axvline(0, color="#AA0000", linestyle="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(plot["变量"])
    ax.invert_yaxis()
    ax.set_xlabel("DataEmbed 系数及 95% 置信区间")
    ax.set_title("核心结果：经营绩效与成本效率显著，TFP 转化整体不显著")
    save_fig(fig, fig_dir / "图5_核心结果系数图.png")

    # 图6：阶段分布
    order = ["基础补课期", "系统嵌入期", "经营改善期", "成本优化期", "转化观察期"]
    dist = diagnosis["阶段诊断2024"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.bar(dist.index, dist.values, color=["#8FAADC", "#A9D18E", "#FFD966", "#F4B183", "#C5A5CF"])
    ax.set_ylabel("企业数量")
    ax.set_title("2024 年样本企业阶段诊断分布")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{int(b.get_height())}", ha="center", va="bottom", fontsize=12)
    ax.set_ylim(0, max(dist.values)+6)
    plt.xticks(rotation=15)
    save_fig(fig, fig_dir / "图6_2024企业阶段分布图.png")

    # 图7：外部示范有效性箱线图
    fig, ax = plt.subplots(figsize=(8, 5.8))
    data = [diagnosis.loc[diagnosis["PolicyDemo_Ever"] == 0, "DataEmbed_w"].dropna(), diagnosis.loc[diagnosis["PolicyDemo_Ever"] == 1, "DataEmbed_w"].dropna()]
    ax.boxplot(data, labels=["未命中示范", "命中示范"], showmeans=True)
    ax.set_ylabel("DataEmbed_w")
    ax.set_title("外部示范标签与数据要素嵌入度分布对比")
    save_fig(fig, fig_dir / "图7_外部示范有效性箱线图.png")

    # 图8：年度趋势
    yearly = panel.groupby("year")[["DataEmbed_w", "ROA_w", "CostRatio_w", "TFP_OP_w"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, label in [("DataEmbed_w", "DataEmbed"), ("ROA_w", "ROA"), ("CostRatio_w", "CostRatio"), ("TFP_OP_w", "TFP_OP")]:
        z = (yearly[col] - yearly[col].mean()) / yearly[col].std()
        ax.plot(yearly["year"], z, marker="o", label=label)
    ax.set_title("核心变量年度趋势（标准化后）")
    ax.set_xlabel("年份")
    ax.set_ylabel("标准化均值")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save_fig(fig, fig_dir / "图8_核心变量年度趋势图.png")

    # 图9：替代指标稳健性
    rob = robust_alt[(robust_alt["因变量"].isin(["ROA_w", "CostRatio_w", "TFP_OP_w"]))].copy()
    rob["名称"] = rob["核心变量"].replace({"DataEmbed_w": "PCA", "DataEmbed_equal_w": "等权", "DataEmbed_entropy_w": "熵权"}) + "-" + rob["因变量"].replace({"ROA_w":"ROA", "CostRatio_w":"Cost", "TFP_OP_w":"TFP"})
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(rob))
    ax.errorbar(rob["系数"], y, xerr=1.96*rob["标准误"], fmt="o", capsize=3, color="#4472C4", ecolor="#888888")
    ax.axvline(0, color="#AA0000", linestyle="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(rob["名称"])
    ax.invert_yaxis()
    ax.set_title("替代指标稳健性：PCA、等权、熵权口径对比")
    ax.set_xlabel("系数及 95% 置信区间")
    save_fig(fig, fig_dir / "图9_替代指标稳健性图.png")

    # 图10：安慰剂检验
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    obs = placebo.loc[placebo["类型"] == "真实样本", "系数"].iloc[0]
    vals = placebo.loc[placebo["类型"] != "真实样本", "系数"].dropna()
    ax.hist(vals, bins=24, alpha=0.75, color="#A6A6A6")
    ax.axvline(obs, color="#C00000", lw=2, label=f"真实系数={obs:.4f}")
    ax.axvline(-obs, color="#C00000", lw=1, linestyle="--")
    ax.set_title("安慰剂检验：年份内随机置换 DataEmbed")
    ax.set_xlabel("置换样本系数")
    ax.set_ylabel("频数")
    ax.legend()
    save_fig(fig, fig_dir / "图10_安慰剂检验分布图.png")

    # 图11：用户操作流程/仪表盘原型
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.axis("off")
    boxes = [
        (0.05, 0.72, 0.22, 0.18, "① 上传/读取数据", "原始面板、映射表\n外部示范名单"),
        (0.39, 0.72, 0.22, 0.18, "② 一键运行脚本", "指标构造\n回归与稳健性"),
        (0.73, 0.72, 0.22, 0.18, "③ 查看仪表盘", "阶段分布\n企业卡片\n风险优先级"),
        (0.05, 0.36, 0.22, 0.18, "④ 选择企业", "按城市、行业、阶段\n外部示范标签筛选"),
        (0.39, 0.36, 0.22, 0.18, "⑤ 生成建议", "基础补课/系统嵌入\n成本优化/转化观察"),
        (0.73, 0.36, 0.22, 0.18, "⑥ 决策落地", "企业项目池\n园区扶持名单\n复盘跟踪"),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#F2F6FF", ec="#4472C4", lw=1.5))
        ax.text(x+w/2, y+h*0.66, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(x+w/2, y+h*0.30, body, ha="center", va="center", fontsize=11)
    for i in [(0.27,0.81,0.39,0.81), (0.61,0.81,0.73,0.81), (0.84,0.72,0.84,0.54), (0.73,0.45,0.61,0.45), (0.39,0.45,0.27,0.45)]:
        ax.annotate("", xy=(i[2], i[3]), xytext=(i[0], i[1]), arrowprops=dict(arrowstyle="->", lw=2, color="#4472C4"))
    ax.text(0.5, 0.15, "系统原型：以 CSV/Excel 为可复现载体，面向企业、园区和评审者提供同一套诊断结果", ha="center", fontsize=13, color="#A85D00", weight="bold")
    save_fig(fig, fig_dir / "图11_用户操作与仪表盘原型图.png")



def make_application_tables(diagnosis: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """生成应用交付层表格：阶段×优先级、城市阶段分布、重点辅导清单。"""
    stage_priority = pd.crosstab(
        diagnosis["阶段诊断2024"],
        diagnosis["优先级"],
        margins=True,
        margins_name="合计"
    ).reset_index()
    city_stage = pd.crosstab(
        diagnosis.get("city", pd.Series("未知", index=diagnosis.index)).fillna("未知"),
        diagnosis["阶段诊断2024"],
        margins=True,
        margins_name="合计"
    ).reset_index().rename(columns={"city": "城市"})
    key_cols = [
        "code6", "short_name", "firm_name", "city", "行业代码D", "阶段诊断2024", "优先级",
        "应用成熟度得分", "赋能转化风险得分", "DataEmbed_w", "ROA_w", "CostRatio_w", "TFP_OP_w",
        "PolicyDemo_Ever", "PolicyDemo_Count_Ever", "诊断建议", "evidence_sentence"
    ]
    key_cols = [c for c in key_cols if c in diagnosis.columns]
    priority = diagnosis[diagnosis["优先级"].astype(str).isin(["优先干预", "重点辅导"])].copy()
    priority = priority.sort_values(["优先级", "赋能转化风险得分", "DataEmbed_w"], ascending=[False, False, False])[key_cols]
    return {"stage_priority": stage_priority, "city_stage": city_stage, "priority_list": priority}


def write_excel_workbook(output_dir: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    """将核心结果表整合为一个可交付工作簿。用户可在本地直接打开查看。"""
    path = output_dir / "数据要素嵌入度诊断与阶段性赋能_复现输出工作簿.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = re.sub(r"[\\/*?:\[\]]", "_", sheet_name)[:31]
            df.to_excel(writer, index=False, sheet_name=safe_name)
    try:
        from openpyxl import load_workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        wb = load_workbook(path)
        header_fill = PatternFill("solid", fgColor="1F4E79")
        header_font = Font(color="FFFFFF", bold=True)
        thin = Side(style="thin", color="D9E2F3")
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = Border(bottom=thin)
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col[:80]:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = min(max(max_len + 2, 10), 38)
            ws.auto_filter.ref = ws.dimensions
        wb.save(path)
    except Exception:
        pass

# ---------------------------------------------------------------------
# 六、主程序
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = project_root()
    parser.add_argument("--input_dir", default=str(INPUT_DIR), help="输入数据目录")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR), help="输出目录")
    parser.add_argument("--placebo_reps", type=int, default=120, help="安慰剂检验置换次数")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    for filename in ("panel.csv", "alias_map.csv"):
        if not (input_dir / filename).is_file():
            parser.error(f"缺少本地输入 {filename}；请检查输入目录，参阅 docs/本地运行.md")
    fig_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    panel_path = input_dir / "panel.csv"
    map_path = input_dir / "alias_map.csv"
    panel = read_csv_smart(panel_path)
    panel["code6"] = panel["code"].apply(ensure_code6)
    mapping = read_csv_smart(map_path)
    alias = build_alias_table(mapping)
    matches = match_external_lists(input_dir, alias)
    upgraded = add_policy_flags(panel, matches)
    diagnosis = classify_stage(upgraded)
    cases = make_case_cards(diagnosis)
    rules = make_decision_rules()
    checklist = make_user_checklist()
    app_tables = make_application_tables(diagnosis)
    regs = run_regressions(upgraded)
    validation = external_validation(diagnosis)
    sensitivity = threshold_sensitivity(upgraded)
    placebo = placebo_test(upgraded, reps=args.placebo_reps)

    # 数据输出
    to_csv(upgraded, output_dir / "01_升级版主面板_含政策示范标签.csv")
    to_csv(diagnosis, output_dir / "02_企业阶段诊断_2024.csv")
    to_csv(cases, output_dir / "03_企业案例卡片_2024.csv")
    to_csv(matches, output_dir / "04_外部示范名单匹配结果.csv")
    to_csv(regs["core"], output_dir / "05_模型结果_核心回归.csv")
    to_csv(regs["lag"], output_dir / "06_模型结果_滞后检验.csv")
    to_csv(regs["mechanism_cost"], output_dir / "07_机制检验_成本与营运通道.csv")
    to_csv(regs["mechanism_added"], output_dir / "08_机制检验_加入成本费用率.csv")
    to_csv(regs["mechanism_summary"], output_dir / "09_机制贡献率说明.csv")
    to_csv(regs["heterogeneity"], output_dir / "10_异质性检验.csv")
    to_csv(regs["robust_alt_index"], output_dir / "11_稳健性_替代指标.csv")
    to_csv(sensitivity, output_dir / "12_稳健性_阶段阈值敏感性.csv")
    to_csv(placebo, output_dir / "13_稳健性_安慰剂检验.csv")
    to_csv(validation, output_dir / "14_外部示范有效性检验.csv")
    to_csv(rules, output_dir / "15_阶段诊断决策规则库.csv")
    to_csv(checklist, output_dir / "16_用户操作清单.csv")
    to_csv(app_tables["stage_priority"], output_dir / "17_阶段与优先级交叉表.csv")
    to_csv(app_tables["city_stage"], output_dir / "18_城市阶段分布表.csv")
    to_csv(app_tables["priority_list"], output_dir / "19_重点辅导与优先干预企业清单.csv")

    write_excel_workbook(output_dir, {
        "01_主面板": upgraded.head(2000),
        "02_阶段诊断2024": diagnosis,
        "03_案例卡片": cases,
        "04_外部示范匹配": matches,
        "05_核心回归": regs["core"],
        "06_滞后检验": regs["lag"],
        "07_机制检验": regs["mechanism_cost"],
        "08_稳健性替代指标": regs["robust_alt_index"],
        "09_阶段阈值敏感性": sensitivity,
        "10_安慰剂检验": placebo,
        "11_外部示范验证": validation,
        "12_规则库": rules,
        "13_用户操作清单": checklist,
        "14_阶段优先级交叉": app_tables["stage_priority"],
        "15_城市阶段分布": app_tables["city_stage"],
        "16_重点辅导清单": app_tables["priority_list"],
    })

    # 图表
    make_figures(upgraded, diagnosis, regs, regs["robust_alt_index"], placebo, fig_dir)

    # 摘要 JSON
    stage_counts = diagnosis["阶段诊断2024"].value_counts().to_dict()
    priority_counts = diagnosis["优先级"].astype(str).value_counts().to_dict()
    summary = {
        "项目名称": "珠三角装备制造企业数据要素嵌入度诊断与阶段性赋能决策支持系统",
        "样本企业数": int(upgraded["code6"].nunique()),
        "企业年份观测数": int(upgraded.shape[0]),
        "主回归期间": "2018-2024",
        "主回归样本量": int(regs["core"].iloc[0]["样本量"]),
        "外部示范匹配条数": int(matches.shape[0]),
        "外部示范命中企业数": int(diagnosis["PolicyDemo_Ever"].sum()),
        "阶段分布": stage_counts,
        "优先级分布": priority_counts,
        "核心回归": regs["core"].to_dict("records"),
        "滞后检验": regs["lag"].to_dict("records"),
        "机制贡献率": regs["mechanism_summary"].to_dict("records"),
        "外部示范有效性检验": validation.to_dict("records"),
        "安慰剂检验置换比例": float(placebo["真实系数绝对值更极端的置换比例"].iloc[0]),
        "输出文件数": len(list(output_dir.glob("*.csv"))) + len(list(fig_dir.glob("*.png"))) + 1,
    }
    with open(output_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
