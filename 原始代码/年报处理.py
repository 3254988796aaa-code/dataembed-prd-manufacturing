# -*- coding: utf-8 -*-
"""
annual_report_datembed_reproduce.py
功能：从公司年报PDF复现论文中的年报文本处理流程
适配：中文文件名年报 + 全角字符兼容 + 99家企业完整别名表
"""
from __future__ import annotations
import argparse
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import unicodedata  # 新增：全角转半角

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    raise ImportError("请先安装 PyMuPDF：pip install pymupdf") from exc

# ============================================================
# 1. 250个关键词词典（严格对齐论文5维度数量）
# ============================================================
TERM_DICT = {
    # 数字化核心词 45个
    "digital_core": [
        "数字化", "数智化", "信息化", "智能化", "数字经济", "人工智能", "AI", "大数据", "云计算",
        "物联网", "IoT", "5G", "区块链", "边缘计算", "数字孪生", "工业互联网", "算法", "算力",
        "模型", "机器学习", "深度学习", "数据驱动", "数字技术", "数字平台", "数字产业化",
        "产业数字化", "知识图谱", "自然语言处理", "计算机视觉", "神经网络", "数据挖掘",
        "数字供应链", "数字营销", "数字治理", "智慧供应链", "线上化", "无纸化", "上云",
        "云原生", "微服务", "中台化", "数字资产", "数据要素", "数字孪生技术", "智能算法"
    ],
    # 智能制造场景词 50个
    "smart_manufacturing": [
        "智能制造", "柔性制造", "智能工厂", "数字化工厂", "智慧工厂", "数字化车间", "智能车间",
        "MES", "APS", "SCADA", "PLC", "DCS", "CNC", "数控", "数控机床", "工业机器人", "机器人",
        "机器视觉", "视觉检测", "智能排产", "智能仓储", "智能物流", "自动化生产线", "自动化产线",
        "设备联网", "设备上云", "预测性维护", "质量追溯", "工艺优化", "黑灯工厂", "无人工厂",
        "智能装备", "智能传感", "传感器", "工业视觉", "缺陷检测", "智能装配", "立体仓库",
        "AGV", "RGV", "智能分拣", "智能调度", "生产节拍", "工艺仿真", "虚拟调试", "数字化工位",
        "智能质检", "在线检测", "实时监控", "远程运维", "远程诊断", "设备健康管理", "产能优化",
        "精益生产", "大规模定制"
    ],
    # 数据/数据治理词 45个
    "data_governance": [
        "数据采集", "数据中台", "数据平台", "数据治理", "数据资产", "数据要素", "数据资源",
        "数据安全", "数据分析", "经营数据", "生产数据", "质量数据", "主数据", "元数据",
        "数据标准", "数据管理", "数据共享", "数据可视化", "数据建模", "数据仓库", "数据库",
        "数据湖", "BI", "看板", "驾驶舱", "数据确权", "数据定价", "数据交易", "数据要素市场化",
        "数据资产化", "数据质量", "数据架构", "数据血缘", "数据生命周期", "数据脱敏",
        "数据加密", "主数据管理", "MDM", "数据集市", "实时数仓", "大数据平台", "数据应用",
        "商业智能", "数据决策", "数据驱动决策"
    ],
    # 系统平台建设词 55个
    "system_platform": [
        "ERP", "MES", "PLM", "SRM", "WMS", "CRM", "SCM", "APS", "QMS", "OA", "EAM", "MOM",
        "工业互联网平台", "云平台", "物联网平台", "协同平台", "管理系统", "信息系统",
        "生产管理系统", "仓储管理系统", "供应链系统", "追溯系统", "检测系统", "控制系统",
        "财务共享", "统一平台", "集成平台", "MES系统", "ERP系统", "PLM系统", "SRM系统",
        "WMS系统", "CRM系统", "SCM系统", "APS系统", "QMS系统", "EAM系统", "生产执行系统",
        "客户关系管理系统", "产品生命周期管理", "先进生产排程", "质量管理系统", "设备管理系统",
        "制造运营管理", "工业APP", "工业软件", "管理驾驶舱", "决策支持系统", "工业控制系统",
        "信息物理系统", "CPS", "数字孪生平台", "工业互联网标识解析", "云制造平台",
        "协同制造平台", "供应链协同平台", "研发设计平台", "仿真平台", "溯源平台"
    ],
    # 落地实施动作词 55个
    "implementation": [
        "研发", "建设", "投产", "上线", "实施", "部署", "运营", "改造", "升级", "集成",
        "验证", "试点", "推广", "应用", "落地", "交付", "验收", "投入使用", "建设完成",
        "正式发布", "量产", "投运", "优化", "迭代", "贯通", "打通", "落地实施", "上线运行",
        "平稳运行", "建成投产", "改造升级", "数字化改造", "智能化改造", "技术改造", "技改",
        "系统集成", "数据打通", "流程打通", "端到端打通", "全链路打通", "业务协同",
        "跨部门协同", "供需协同", "产销协同", "业财一体化", "管控一体化", "两化融合",
        "数字化转型", "智能化升级", "数智化转型", "信息化建设", "数字化建设", "平台建设",
        "体系建设", "能力建设", "场景落地", "应用落地", "试点示范", "标杆建设", "复制推广",
        "全面推广", "深化应用", "持续优化", "提质增效", "降本增效", "节能减排", "精益管理",
        "精细化管理", "标准化管理", "数字化运营", "智能化运营"
    ],
}

CORE_GROUPS = ["digital_core", "smart_manufacturing", "data_governance", "system_platform"]
FEATURE_GROUPS = ["digital_core", "smart_manufacturing", "data_governance", "system_platform", "implementation"]

LANDMARK_TERMS = [
    "数字化转型", "智能制造", "工业互联网", "数据中台", "MES", "ERP", "PLM", "自动化生产线",
    "数字化工厂", "智能工厂", "数据治理", "数据资产", "工业互联网平台"
]


# ============================================================
# 2. 工具函数：名称归一化 + 别名加载（新增全角转半角）
# ============================================================
def normalize_name(s: str) -> str:
    """名称归一化：全角转半角、去空格、去特殊符号、转小写，用于匹配"""
    if pd.isna(s):
        return ""
    s = str(s)
    # 全角转半角（核心修复：解决深天马Ａ、深纺织Ａ的匹配问题）
    s = unicodedata.normalize('NFKC', s)
    # 去所有空格
    s = re.sub(r"\s+", "", s)
    # 统一括号
    s = s.replace("（", "(").replace("）", ")")
    # 去括号及内容
    s = re.sub(r"\(.*?\)", "", s)
    # 去其他特殊符号
    s = re.sub(r"[：:·•,，。、“”\"'\[\]【】/\\;；]", "", s)
    # 转小写
    return s.lower()


def load_alias_map(alias_file: str | None) -> dict[str, str]:
    """加载alias_map.csv，构建「所有别名→股票代码」的映射"""
    if not alias_file:
        return {}
    df = pd.read_csv(alias_file, dtype=str)
    out = {}
    for _, r in df.iterrows():
        code = str(r.get("stock_code", "")).zfill(6)
        if not code:
            continue
        # 主名称加入映射
        main_name = str(r.get("firm_name", "")).strip()
        if main_name and main_name.lower() != "nan":
            out[normalize_name(main_name)] = code
        # 别名加入映射
        aliases = str(r.get("aliases", "")).replace("，", ";").replace(",", ";").split(";")
        for alias in aliases:
            alias = alias.strip()
            if alias and alias.lower() != "nan":
                out[normalize_name(alias)] = code
    return out


def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> tuple[str, int]:
    chunks = []
    with fitz.open(pdf_path) as doc:
        n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)
        for i in range(n_pages):
            chunks.append(doc[i].get_text("text"))
    return "\n".join(chunks), n_pages


def clean_report_text(text: str) -> str:
    text = re.sub(r"第\s*\d+\s*页\s*(?:共\s*\d+\s*页)?", " ", text)
    text = re.sub(r"\b\d+\s*/\s*\d+\b", " ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[■□▪●◆►▷]", " ", text)
    return text.strip()


def parse_filename(pdf_name: str, alias_map: dict[str, str]) -> tuple[int | None, str, str]:
    """
    从中文文件名中提取：年份、股票代码、原始企业名
    支持格式：
    - 美的集团：2017年年度报告.pdf
    - TCL 科技：2019年年度报告.pdf
    - 万和电气：2022年年度报告（更新后）.pdf
    - 深天马Ａ：2017年年度报告.pdf
    """
    name = pdf_name.replace(".pdf", "").replace(".PDF", "")

    # 1. 提取年份
    year_match = re.search(r"(20[0-3]\d)", name)
    year = int(year_match.group(1)) if year_match else None

    # 2. 提取企业名称（去掉年份、年报相关后缀、括号备注）
    raw_name = name
    # 去掉年份及后面的内容
    if year_match:
        raw_name = raw_name[:year_match.start()]
    # 去掉常见后缀
    raw_name = re.sub(r"[：:].*$", "", raw_name)  # 去掉冒号及后面
    raw_name = re.sub(r"年度报告.*$", "", raw_name)
    raw_name = re.sub(r"年报.*$", "", raw_name)
    raw_name = re.sub(r"全文.*$", "", raw_name)
    raw_name = re.sub(r"（更新后）.*$", "", raw_name)
    raw_name = re.sub(r"\(更新后\).*$", "", raw_name)
    raw_name = raw_name.strip()

    # 3. 归一化后匹配别名表
    norm_name = normalize_name(raw_name)
    code = alias_map.get(norm_name, "")

    # 4. 兜底：如果精确匹配不到，尝试包含匹配（取最长匹配）
    if not code:
        best_code, best_len = "", 0
        for alias_norm, c in alias_map.items():
            if len(alias_norm) >= 2 and (alias_norm in norm_name or norm_name in alias_norm):
                if len(alias_norm) > best_len:
                    best_len = len(alias_norm)
                    best_code = c
        code = best_code

    return year, code, raw_name


# ============================================================
# 3. 词频统计与证据等级
# ============================================================
def count_terms(text: str, terms: list[str]) -> tuple[int, str]:
    lower_text = text.lower()
    counts = Counter()
    for term in terms:
        if not term:
            continue
        if re.search(r"[A-Za-z]", term):
            c = lower_text.count(term.lower())
        else:
            c = text.count(term)
        if c:
            counts[term] = c
    total = sum(counts.values())
    top_terms = ";".join([f"{k}:{v}" for k, v in counts.most_common(15)])
    return total, top_terms


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？；;!?]\s*", text)
    return [s.strip() for s in parts if len(s.strip()) >= 15]


def sentence_score(sent: str) -> float:
    score = 0.0
    for group in CORE_GROUPS:
        hit = any(t.lower() in sent.lower() if re.search(r"[A-Za-z]", t) else t in sent for t in TERM_DICT[group])
        if hit:
            score += 2.0
    if any(t in sent for t in LANDMARK_TERMS):
        score += 3.0
    if any(t in sent for t in TERM_DICT["implementation"]):
        score += 1.5
    return score


def best_evidence_sentence(text: str) -> str:
    sents = split_sentences(text)
    if not sents:
        return ""
    return max(sents, key=sentence_score)[:180]


def evidence_level(core_total: int, best_sent: str) -> int:
    landing_flag = any(t in best_sent for t in LANDMARK_TERMS) or any(
        t in best_sent for t in TERM_DICT["implementation"])
    if (core_total >= 250 and landing_flag) or (core_total >= 80 and landing_flag):
        return 4
    if core_total >= 40:
        return 3
    if core_total >= 10:
        return 2
    return 1


# ============================================================
# 4. 单PDF处理
# ============================================================
def process_one_pdf(pdf_path: Path, alias_map: dict[str, str], max_pages: int | None) -> dict:
    raw_text, pages_used = extract_pdf_text(pdf_path, max_pages=max_pages)
    text = clean_report_text(raw_text)
    year, stock_code, raw_firm_name = parse_filename(pdf_path.name, alias_map)

    row = {
        "source_file": pdf_path.name,
        "raw_firm_name": raw_firm_name,
        "stock_code": stock_code,
        "year": year,
        "pages_used": pages_used,
        "text_len": len(text),
    }

    for group in FEATURE_GROUPS:
        cnt, top = count_terms(text, TERM_DICT[group])
        row[f"{group}_count"] = cnt
        row[f"{group}_density"] = cnt / max(len(text), 1) * 10000
        row[f"{group}_top_terms"] = top

    core_total = sum(row[f"{g}_count"] for g in CORE_GROUPS)
    best_sent = best_evidence_sentence(text)
    row["core_keyword_total"] = core_total
    row["evidence_sentence"] = best_sent
    row["evidence_level"] = evidence_level(core_total, best_sent)

    return row


# ============================================================
# 5. 标准化与指标构建
# ============================================================
def zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    sd = x.std(skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.mean(skipna=True)) / sd


def entropy_weight(X: pd.DataFrame) -> np.ndarray:
    X = X.to_numpy(dtype=float)
    n, m = X.shape
    X = X - X.min(axis=0) + 1e-6
    P = X / X.sum(axis=0, keepdims=True)
    E = -np.sum(P * np.log(P), axis=0) / np.log(n)
    D = 1 - E
    W = D / D.sum()
    return W


def add_datembed_indexes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [f"{g}_density" for g in FEATURE_GROUPS] + ["evidence_level"]
    out = df.copy()

    X_std = pd.DataFrame({c: zscore(out[c]) for c in feature_cols}).fillna(0.0)
    X_mat = X_std.to_numpy(dtype=float)

    if len(out) < 2:
        out["DataEmbed"] = np.nan
        out["DataEmbed_equal"] = np.nan
        out["DataEmbed_entropy"] = np.nan
        loadings = pd.DataFrame({"feature": feature_cols, "pca_loading": np.nan, "explained_variance_ratio": np.nan})
        return out, loadings

    cov = np.cov(X_mat, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    pc1_val = eigvals[order[0]]
    pc1_vec = eigvecs[:, order[0]]

    if np.nansum(pc1_vec) < 0:
        pc1_vec = -pc1_vec

    score_pca = X_mat @ pc1_vec
    out["DataEmbed"] = score_pca
    explained = pc1_val / eigvals.sum() if eigvals.sum() != 0 else np.nan

    loadings = pd.DataFrame({
        "feature": feature_cols,
        "pca_loading": pc1_vec,
        "explained_variance_ratio": explained,
    })

    out["DataEmbed_equal"] = X_std.mean(axis=1)
    w_entropy = entropy_weight(X_std)
    out["DataEmbed_entropy"] = X_std.to_numpy() @ w_entropy

    out["DataEmbed_z"] = zscore(out["DataEmbed"])
    out["DataEmbed_equal_z"] = zscore(out["DataEmbed_equal"])
    out["DataEmbed_entropy_z"] = zscore(out["DataEmbed_entropy"])

    return out, loadings


# ============================================================
# 6. 主程序（已配置好路径，直接运行）
# ============================================================
def main():
    # ========== 路径配置，按需修改 ==========
    pdf_dir = Path(r"公司年报PDF")  # 年报PDF文件夹
    out_dir = Path(r"年报处理结果")  # 输出结果文件夹
    alias_file = r"alias_map.csv"  # 99家企业完整别名表
    max_pages = None  # 不限制页数
    # =======================================

    out_dir.mkdir(parents=True, exist_ok=True)
    alias_map = load_alias_map(alias_file)
    print(f"加载别名映射表，共 {len(alias_map)} 个别名条目，覆盖99家企业")

    # 修复Windows大小写重复匹配
    pdf_set = set()
    for pattern in ["*.pdf", "*.PDF"]:
        for f in pdf_dir.glob(pattern):
            pdf_set.add(f.resolve())
    pdfs = sorted(pdf_set)

    if not pdfs:
        raise FileNotFoundError(f"未在 {pdf_dir} 找到 PDF 文件")

    print(f"找到 {len(pdfs)} 个年报PDF，开始处理...\n")
    records = []
    unmatched = []

    for i, pdf in enumerate(pdfs, 1):
        try:
            row = process_one_pdf(pdf, alias_map, max_pages)
            records.append(row)
            if not row["stock_code"]:
                unmatched.append(row["source_file"])
            print(
                f"[{i}/{len(pdfs)}] {row['stock_code'] if row['stock_code'] else '未匹配'} | {row['raw_firm_name']} {row['year']}年")
        except Exception as exc:
            records.append({"source_file": pdf.name, "quality_note": f"failed: {exc}"})
            print(f"[{i}/{len(pdfs)}] 处理失败: {pdf.name} | 错误: {exc}")

    raw = pd.DataFrame(records)
    raw.to_csv(out_dir / "annual_report_text_features_raw.csv", index=False, encoding="utf-8-sig")

    # 过滤掉未匹配到代码的记录
    valid = raw[raw["stock_code"].astype(str).str.len() == 6].copy()
    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"总PDF数：{len(pdfs)}")
    print(f"成功匹配企业：{len(valid)} 条")
    print(f"未匹配企业：{len(unmatched)} 个")
    if unmatched:
        print("未匹配文件名示例（前10个）：")
        for f in unmatched[:10]:
            print(f"  - {f}")

    # 去重：同一企业-年份保留文本更长、核心词更多的一份
    panel = valid.copy()
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce").astype("Int64")
    panel = panel.sort_values(
        ["stock_code", "year", "text_len", "core_keyword_total"],
        ascending=[True, True, False, False]
    )
    panel = panel.drop_duplicates(["stock_code", "year"], keep="first")

    # 生成三种DataEmbed指标
    panel, loadings = add_datembed_indexes(panel)

    panel.to_csv(out_dir / "datembed_panel.csv", index=False, encoding="utf-8-sig")
    loadings.to_csv(out_dir / "datembed_pca_loadings.csv", index=False, encoding="utf-8-sig")

    print(f"\n最终面板：{panel.shape[0]} 条企业-年份观测，共 {panel['stock_code'].nunique()} 家企业")
    print(f"结果已输出到：{out_dir}")


if __name__ == "__main__":
    main()