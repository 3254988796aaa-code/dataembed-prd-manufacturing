# -*- coding: utf-8 -*-
"""
数据要素嵌入度交互式诊断仪表盘
运行方式：streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# ===================== 0. 页面全局配置 =====================
st.set_page_config(
    page_title="珠三角装备制造企业数据要素嵌入度诊断系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Plotly中文字体配置
PLOTLY_FONT = dict(family="SimHei, Microsoft YaHei, PingFang SC", size=12)


# ===================== 1. 工具函数：双向固定效应 =====================
def twoway_residualize(df, cols, entity="code6", time="year"):
    out = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].astype(float)
        out[c] = s - s.groupby(df[entity]).transform("mean") - s.groupby(df[time]).transform("mean") + s.mean()
    return out


def run_fe_reg(data, dep, x, controls, entity="code6", time="year"):
    cols = [dep, x, entity, time] + controls
    d = data[cols].dropna().copy()
    if len(d) < 30:
        return None

    all_vars = [dep, x] + controls
    r = twoway_residualize(d, all_vars, entity, time)
    y = r[dep]
    X = r[[x] + controls]

    try:
        model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": d[entity]})
        result = {
            "变量": [x] + controls,
            "系数": [round(model.params[v], 4) for v in [x] + controls],
            "标准误": [round(model.bse[v], 4) for v in [x] + controls],
            "P值": [round(model.pvalues[v], 4) for v in [x] + controls],
            "显著性": []
        }
        for p in result["P值"]:
            if p < 0.01:
                result["显著性"].append("***")
            elif p < 0.05:
                result["显著性"].append("**")
            elif p < 0.1:
                result["显著性"].append("*")
            else:
                result["显著性"].append("")
        result["样本量"] = int(model.nobs)
        result["组内R²"] = round(model.rsquared, 4)
        return pd.DataFrame(result), model
    except:
        return None, None


# ===================== 2. 数据加载（缓存加速） =====================
@st.cache_data
def load_data():
    df = pd.read_csv("C://Users//HP//Desktop//DataEmbed_reproducible_code//raw_data//数据分析//输入数据//panel.csv", dtype=str)

    # 兼容code/code6
    if "code6" in df.columns:
        df["code6"] = df["code6"].astype(str).str.zfill(6)
    elif "code" in df.columns:
        df["code6"] = df["code"].astype(str).str.zfill(6)

    # 数值列转换
    num_cols = [
        "year", "DataEmbed_w", "DataEmbed_equal_w", "DataEmbed_entropy_w",
        "ROA_w", "OperatingMargin_w", "NetMargin_w", "CostRatio_w", "CostEfficiency_w",
        "TFP_OP_w", "TFP_LP_w", "Size_w", "Lev_w", "FixRatio_w",
        "EmployeeSize_w", "RDIntensity_w", "ln_GDPpc_w", "ln_FiscalTech_w", "ln_DigitalInfra_w",
        "evidence_level", "HighEvidence", "HighSize", "HighRD", "HighDigitalInfra"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 填充缺失的分类列
    if "city" not in df.columns:
        df["city"] = "未知"
    if "行业代码D" not in df.columns:
        df["行业代码D"] = "通用设备"
    if "short_name" not in df.columns:
        df["short_name"] = df["code6"]

    return df


df = load_data()

# ===================== 3. 侧边栏筛选器 =====================
st.sidebar.title("🔍 筛选条件")

# 年份范围
year_min = int(df["year"].min())
year_max = int(df["year"].max())
year_range = st.sidebar.slider("年份区间", year_min, year_max, (year_min, year_max))

# 城市筛选
cities = sorted(df["city"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect("城市", cities, default=cities)

# 行业筛选
industries = sorted(df["行业代码D"].dropna().unique().tolist())
selected_inds = st.sidebar.multiselect("行业", industries, default=industries)

# 企业规模分组
size_group = st.sidebar.radio("企业规模", ["全部", "大规模", "小规模"])

# 应用筛选
filtered = df[
    (df["year"] >= year_range[0]) & (df["year"] <= year_range[1]) &
    (df["city"].isin(selected_cities)) &
    (df["行业代码D"].isin(selected_inds))
    ].copy()

if size_group == "大规模":
    filtered = filtered[filtered["HighSize"] == 1]
elif size_group == "小规模":
    filtered = filtered[filtered["HighSize"] == 0]

st.sidebar.caption(f"当前筛选样本：{filtered['code6'].nunique()} 家企业，共 {len(filtered)} 条观测")

# ===================== 4. 主页面Tab =====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 总览看板", "🏭 企业详情", "📊 实证实验室", "💡 策略模拟器", "📖 方法说明"
])

# ---------------- Tab1：总览看板 ----------------
with tab1:
    st.header("📈 样本整体概览")

    # 核心指标卡片
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("样本企业数", f"{filtered['code6'].nunique()} 家")
    with col2:
        st.metric("平均DataEmbed", f"{filtered['DataEmbed_w'].mean():.4f}")
    with col3:
        st.metric("平均ROA", f"{filtered['ROA_w'].mean():.4f}")
    with col4:
        st.metric("平均成本费用率", f"{filtered['CostRatio_w'].mean():.4f}")

    st.divider()

    # 两行图表
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("DataEmbed年度趋势")
        yearly_de = filtered.groupby("year")["DataEmbed_w"].mean().reset_index()
        fig = px.line(yearly_de, x="year", y="DataEmbed_w", markers=True,
                      title="数据要素嵌入度年度均值变化")
        fig.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

    with row1_col2:
        st.subheader("城市维度分布")
        city_de = filtered.groupby("city")["DataEmbed_w"].mean().reset_index().sort_values("DataEmbed_w",
                                                                                           ascending=False)
        fig = px.bar(city_de, x="city", y="DataEmbed_w", color="DataEmbed_w",
                     color_continuous_scale="Blues", title="各城市平均DataEmbed排名")
        fig.update_layout(font=PLOTLY_FONT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("DataEmbed vs ROA 散点图")
        latest = filtered[filtered["year"] == year_max]
        fig = px.scatter(latest, x="DataEmbed_w", y="ROA_w",
                         color="city", hover_data=["short_name", "city"],
                         title=f"{year_max}年 DataEmbed 与 ROA 散点分布")
        fig.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

    with row2_col2:
        st.subheader("证据等级分布")
        evidence_dist = latest["evidence_level"].value_counts().reset_index()
        evidence_dist.columns = ["证据等级", "企业数"]
        fig = px.pie(evidence_dist, values="企业数", names="证据等级",
                     title="年报数字化证据等级分布", hole=0.4)
        fig.update_layout(font=PLOTLY_FONT)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab2：企业详情 ----------------
with tab2:
    st.header("🏭 企业级详细诊断")

    # 选择企业
    firm_list = sorted(filtered["short_name"].dropna().unique().tolist())
    selected_firm = st.selectbox("选择企业", firm_list)

    firm_data = filtered[filtered["short_name"] == selected_firm].sort_values("year")

    if len(firm_data) == 0:
        st.warning("该企业无匹配数据")
    else:
        # 企业基础信息
        latest_firm = firm_data.iloc[-1]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("股票代码", latest_firm["code6"])
        with col2:
            st.metric("所在城市", latest_firm["city"])
        with col3:
            st.metric("最新DataEmbed", f"{latest_firm['DataEmbed_w']:.4f}")
        with col4:
            st.metric("证据等级", f"{int(latest_firm['evidence_level'])} 级")

        st.divider()

        # 历年趋势图
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=firm_data["year"], y=firm_data["DataEmbed_w"],
                                     mode="lines+markers", name="DataEmbed", line=dict(color="#1f77b4")))
            fig.update_layout(title="数据要素嵌入度历年变化", font=PLOTLY_FONT, xaxis_title="年份")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=firm_data["year"], y=firm_data["ROA_w"],
                                     mode="lines+markers", name="ROA", line=dict(color="#2ca02c")))
            fig.add_trace(go.Scatter(x=firm_data["year"], y=firm_data["CostRatio_w"],
                                     mode="lines+markers", name="成本费用率", line=dict(color="#ff7f0e"), yaxis="y2"))
            fig.update_layout(
                title="经营绩效与成本效率历年变化",
                font=PLOTLY_FONT, xaxis_title="年份",
                yaxis=dict(title="ROA"),
                yaxis2=dict(title="成本费用率", overlaying="y", side="right")
            )
            st.plotly_chart(fig, use_container_width=True)

        # 详细指标表
        st.subheader("历年详细指标")
        show_cols = ["year", "DataEmbed_w", "ROA_w", "CostRatio_w", "TFP_OP_w",
                     "Size_w", "Lev_w", "RDIntensity_w", "evidence_level"]
        show_cols = [c for c in show_cols if c in firm_data.columns]
        st.dataframe(firm_data[show_cols].round(4), use_container_width=True, hide_index=True)

# ---------------- Tab3：实证实验室 ----------------
with tab3:
    st.header("📊 双向固定效应实证实验室")
    st.caption("动态调整参数，实时计算回归结果，验证结论稳健性")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("参数设置")
        dep_var = st.selectbox("因变量",
                               ["ROA_w", "OperatingMargin_w", "NetMargin_w", "CostRatio_w", "TFP_OP_w"],
                               index=0)
        x_var = st.selectbox("核心解释变量",
                             ["DataEmbed_w", "DataEmbed_equal_w", "DataEmbed_entropy_w"],
                             index=0)

        st.write("控制变量")
        all_controls = ["Size_w", "Lev_w", "FixRatio_w", "EmployeeSize_w", "RDIntensity_w",
                        "ln_GDPpc_w", "ln_FiscalTech_w", "ln_DigitalInfra_w"]
        default_controls = [c for c in all_controls if c in filtered.columns]
        selected_controls = st.multiselect("选择控制变量", default_controls, default=default_controls)

        cluster = st.radio("聚类标准误", ["企业层面聚类", "不聚类"], index=0)
        run_btn = st.button("🚀 运行回归", type="primary", use_container_width=True)

    with col2:
        st.subheader("回归结果")
        if run_btn:
            res_df, model = run_fe_reg(filtered, dep_var, x_var, selected_controls)
            if res_df is not None:
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.caption(f"样本量：{res_df['样本量'].iloc[0]}  |  组内R²：{res_df['组内R²'].iloc[0]}")

                # 系数图
                fig = px.bar(res_df, x="变量", y="系数", error_y="标准误",
                             color="显著性",
                             color_discrete_map={"***": "#d62728", "**": "#ff7f0e", "*": "#ffbb78", "": "#9467bd"},
                             title="回归系数及置信区间")
                fig.add_hline(y=0, line_dash="dash", line_color="black")
                fig.update_layout(font=PLOTLY_FONT)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("样本量不足或数据缺失，无法运行回归")
        else:
            st.info("👈 左侧设置参数后点击运行")

# ---------------- Tab4：策略模拟器 ----------------
with tab4:
    st.header("💡 数字化转型策略反事实模拟器")
    st.caption("调整转型强度，实时测算绩效改善效果")

    latest_data = filtered[filtered["year"] == year_max].copy()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("策略参数")
        lift_std = st.slider("DataEmbed提升幅度（标准差）", 0.0, 2.0, 0.5, 0.1)
        de_std = df["DataEmbed_w"].std()
        lift_abs = lift_std * de_std
        st.metric("绝对提升幅度", f"{lift_abs:.4f}")

        # 基准系数（用全样本基准回归）
        base_roa_coef = 0.0302
        base_cost_coef = -0.0341

        # 计算平均效果
        avg_roa_change = base_roa_coef * lift_abs
        avg_cost_change = base_cost_coef * lift_abs
        avg_roa_pct = avg_roa_change / latest_data["ROA_w"].mean() * 100
        avg_cost_pct = avg_cost_change / latest_data["CostRatio_w"].mean() * 100

        st.divider()
        st.metric("ROA平均提升", f"{avg_roa_change:.4f}", f"{avg_roa_pct:.2f}%")
        st.metric("成本费用率平均下降", f"{abs(avg_cost_change):.4f}", f"{abs(avg_cost_pct):.2f}%")

    with col2:
        st.subheader("企业分布变化")
        latest_data["ROA_反事实"] = latest_data["ROA_w"] + avg_roa_change

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=latest_data["ROA_w"], name="转型前", opacity=0.6, nbinsx=20))
        fig.add_trace(go.Histogram(x=latest_data["ROA_反事实"], name="转型后", opacity=0.6, nbinsx=20))
        fig.update_layout(
            title="ROA分布：转型前后对比",
            barmode="overlay",
            font=PLOTLY_FONT,
            xaxis_title="ROA",
            yaxis_title="企业数"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 动态路径
    st.subheader("未来5年动态演化路径")
    years = [2025, 2026, 2027, 2028, 2029]
    build_period = st.slider("建设周期（年）", 1, 5, 3)

    release = [min((i + 1) / build_period, 1.0) for i in range(5)]
    roa_path = [avg_roa_change * r for r in release]
    cost_path = [avg_cost_change * r for r in release]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=roa_path, mode="lines+markers", name="ROA累计提升", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=years, y=cost_path, mode="lines+markers", name="成本费用率累计下降", line=dict(width=3),
                             yaxis="y2"))
    fig.update_layout(
        title="转型绩效释放动态路径",
        font=PLOTLY_FONT, xaxis_title="年份",
        yaxis=dict(title="ROA累计提升"),
        yaxis2=dict(title="成本费用率累计变化", overlaying="y", side="right")
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Tab5：方法说明 ----------------
with tab5:
    st.header("📖 方法与数据说明")

    st.subheader("1. 指标构建")
    st.markdown("""
    - **DataEmbed 数据要素嵌入度**：基于年报文本挖掘，从数字化核心、智能制造、数据治理、系统平台、落地实施5个维度共250个关键词构建，采用PCA主成分合成，同时提供等权、熵权两种替代口径做稳健性检验。
    - **证据等级**：结合关键词密度与落地表述，将企业数字化披露分为1-4级，等级越高代表实质性落地证据越充分。
    """)

    st.subheader("2. 计量模型")
    st.markdown("""
    采用双向固定效应模型（TWFE）控制个体与时间固定效应，标准误聚类到企业层面：
    $$
    Y_{it} = \\beta \\times DataEmbed_{it} + \\gamma \\times Controls_{it} + \\alpha_i + \\delta_t + \\epsilon_{it}
    $$
    其中 $\\alpha_i$ 为个体固定效应，$\\delta_t$ 为时间固定效应。
    """)

    st.subheader("3. 数据来源")
    st.markdown("""
    - 文本数据：上市公司年度报告PDF，全文本提取与关键词匹配
    - 财务数据：上市公司公开财务指标
    - 城市数据：各地市统计年鉴、国民经济和社会发展统计公报
    """)

    st.subheader("4. 系统说明")
    st.caption("本系统基于 Streamlit + Plotly 全Python实现，纯文件架构，无需数据库，支持增量数据一键更新。")

# 页脚
st.divider()
st.caption("珠三角装备制造企业数据要素嵌入度诊断与阶段性赋能系统 | 全Python实现")