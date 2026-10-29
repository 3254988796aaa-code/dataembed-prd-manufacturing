# -*- coding: utf-8 -*-
"""
整合生成最终面板数据.py (升级版)
修复：笛卡尔积行数爆炸问题、证券代码前导零截断问题
升级：插值仅内插、城市宏观缺失保持NaN、完善注释与输出
"""
import pandas as pd
import numpy as np
import re
import os
from _local_data import local_input, local_output, load_deflator, load_stock_city_data
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ============================================================
# 0. 工具函数
# ============================================================
def winsorize(s: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lower = s.quantile(q_low)
    upper = s.quantile(q_high)
    return s.clip(lower=lower, upper=upper)

def ensure_code6(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(".0", "")
    s = re.sub(r"\D", "", s)
    return s.zfill(6) if s else ""

# ============================================================
# 1. 从本地私有文件读取企业股票代码-城市映射
# ============================================================
STOCK_CITY_DATA = load_stock_city_data()

stock_city_df = pd.DataFrame(STOCK_CITY_DATA, columns=["firm_name", "code", "city"])
stock_city_df["code"] = stock_city_df["code"].apply(ensure_code6)
stock_city_map = stock_city_df[["code", "city", "firm_name"]].drop_duplicates("code")

# ============================================================
# 2. 价格平减指数
# ============================================================
PPI_DEFLATOR = load_deflator("ppi_deflator.csv")

# ============================================================
# 3. Windows本地路径配置
# ============================================================
PATH_TFP = local_input("TFP_2017-2024.xlsx", required=True)
PATH_FINANCE = local_input("上市公司财务数据_最终表02.xlsx", required=True)
PATH_TFP_RAW = local_input("计算TFP所需的原始财务数据面板.xlsx", required=True)
PATH_DATEMBED = local_input("datembed_panel.csv", required=True)
PATH_CITY_MACRO = local_input("城市宏观数据.xlsx", required=False)
OUTPUT_CSV = local_output("panel.csv")

# ============================================================
# 4. 读取各类基础数据表（显式dtype与去重）
# ============================================================
# TFP
tfp_df = pd.read_excel(PATH_TFP, dtype={'stkcd': str})
tfp_df = tfp_df.rename(columns={'stkcd': 'code', '股票简称': 'short_name'})
tfp_cols = ['code', 'short_name', 'year', 'TFP_OP', 'TFP_LP', 'TFP_OLS', 'TFP_FE', 'TFP_GMM']
tfp_df = tfp_df[tfp_cols]
tfp_df['code'] = tfp_df['code'].apply(ensure_code6)
tfp_df = tfp_df.drop_duplicates(subset=['code', 'year'], keep='last')  # keep='last'可保留最新数据

# 财务数据
finance_df = pd.read_excel(PATH_FINANCE, dtype={'证券代码': str})
finance_df = finance_df.rename(columns={'证券代码': 'code', '时间': 'year'})
finance_df['code'] = finance_df['code'].apply(ensure_code6)
finance_df = finance_df.drop_duplicates(subset=['code', 'year'], keep='last')

# TFP原始投入产出
tfp_raw_df = pd.read_excel(PATH_TFP_RAW, dtype={'证券代码': str})
tfp_raw_df = tfp_raw_df.rename(columns={'证券代码': 'code', '时间': 'year'})
tfp_raw_df['code'] = tfp_raw_df['code'].apply(ensure_code6)
tfp_raw_cols = ['code', 'year', '营业成本_元', '固定资产净额_元', '员工人数_个', '营业收入_元', '折旧摊销']
tfp_raw_df = tfp_raw_df[tfp_raw_cols]
tfp_raw_df = tfp_raw_df.drop_duplicates(subset=['code', 'year'], keep='last')

# 年报DataEmbed面板
annual_report_df = pd.read_csv(PATH_DATEMBED, dtype=str)
annual_report_df = annual_report_df.rename(columns={'stock_code': 'code'})
annual_report_df['code'] = annual_report_df['code'].apply(ensure_code6)
annual_report_df['year'] = pd.to_numeric(annual_report_df['year'], errors='coerce')
num_cols = [c for c in annual_report_df.columns if c.endswith('_count') or c.endswith('_density') or c in ['core_keyword_total', 'evidence_level', 'DataEmbed', 'DataEmbed_equal', 'DataEmbed_entropy']]
for col in num_cols:
    annual_report_df[col] = pd.to_numeric(annual_report_df[col], errors='coerce')
# 去除空值年份并去重
annual_report_df = annual_report_df.dropna(subset=['year']).drop_duplicates(subset=['code', 'year'], keep='last')

# ============================================================
# 5. 城市宏观数据读取（缺失保持NaN，避免人为填充小值）
# ============================================================
city_macro_df = pd.DataFrame()
city_macro_available = False
if os.path.exists(PATH_CITY_MACRO):
    try:
        raw_city = pd.read_excel(PATH_CITY_MACRO, header=1)
        raw_city = raw_city.rename(columns={raw_city.columns[0]: 'city'})
        raw_city['city'] = raw_city['city'].astype(str).str.strip()
        new_columns = ['city']
        for col in raw_city.columns[1:]:
            col_str = str(col).strip()
            year_match = re.search(r'(20[0-3]\d)', col_str)
            new_columns.append(year_match.group(1) if year_match else col_str)
        raw_city.columns = new_columns
        year_cols = [str(y) for y in range(2017, 2025)]

        def melt_indf(df, val_name):
            blk = df[['city'] + year_cols].copy()
            longdf = blk.melt(id_vars='city', var_name='year', value_name=val_name)
            longdf['year'] = pd.to_numeric(longdf['year'], errors='coerce')
            # 数值列转换，不再填充极小值
            longdf[val_name] = pd.to_numeric(longdf[val_name], errors='coerce')
            return longdf.dropna(subset=['year'])

        gdp_df = melt_indf(raw_city, "GDP")
        gdp_pc_df = melt_indf(raw_city, "GDPpc")
        infra_df = melt_indf(raw_city, "DigitalInfra")
        tech_df = melt_indf(raw_city, "FiscalTech")

        city_macro_df = gdp_df
        for subdf in [gdp_pc_df, infra_df, tech_df]:
            city_macro_df = pd.merge(city_macro_df, subdf, on=['city', 'year'], how='outer')
        # 仅对正值取对数，缺失保持NaN
        for col in ['GDP', 'GDPpc', 'FiscalTech', 'DigitalInfra']:
            city_macro_df[f'ln_{col}'] = np.log(city_macro_df[col].clip(lower=1e-6))  # 小正值防止log(0)，但缺失仍为NaN
            # 若仍想避免负无穷，可进一步处理：city_macro_df.loc[city_macro_df[col] <= 0, f'ln_{col}'] = np.nan
        city_macro_df['year'] = city_macro_df['year'].astype(int)

        # 增加城市数据去重保护
        city_macro_df = city_macro_df.drop_duplicates(subset=['city', 'year'], keep='last')
        city_macro_available = True
        print(f"✅ 城市宏观文件读取成功，城市数：{city_macro_df['city'].nunique()}")
    except Exception as e:
        print(f"⚠️ 城市文件读取失败：{e}，跳过城市宏观变量")
else:
    print(f"⚠️ 未找到{PATH_CITY_MACRO}，跳过城市宏观数据")

# ============================================================
# 6. 多表合并
# ============================================================
merged_df = pd.merge(tfp_df, finance_df, on=['code', 'year'], how='left')
merged_df = pd.merge(merged_df, tfp_raw_df, on=['code', 'year'], how='left')
merged_df = pd.merge(merged_df, annual_report_df, on=['code', 'year'], how='left')
merged_df = pd.merge(merged_df, stock_city_map, on=['code'], how='left')

if city_macro_available and 'city' in city_macro_df.columns:
    merged_df = pd.merge(merged_df, city_macro_df, on=['city', 'year'], how='left')

merged_df['code6'] = merged_df['code'].apply(ensure_code6)

# ============================================================
# 7. 财务指标计算
# ============================================================
merged_df['ppi'] = merged_df['year'].map(PPI_DEFLATOR)
nominal_cols = ['营业总收入', '营业总成本', '营业利润', '净利润', '资产总计', '负债合计', '研发投入金额']
for col in nominal_cols:
    if col in merged_df.columns:
        merged_df[f'{col}_real'] = merged_df[col] / merged_df['ppi']

if all(x in merged_df.columns for x in ['净利润','资产总计']):
    merged_df['ROA'] = merged_df['净利润'] / merged_df['资产总计']
if all(x in merged_df.columns for x in ['营业利润','营业总收入','净利润','营业总成本']):
    merged_df['OperatingMargin'] = merged_df['营业利润'] / merged_df['营业总收入']
    merged_df['NetMargin'] = merged_df['净利润'] / merged_df['营业总收入']
    merged_df['CostRatio'] = merged_df['营业总成本'] / merged_df['营业总收入']
    merged_df['CostEfficiency'] = merged_df['营业总收入'] / merged_df['营业总成本']

turnover_cols = ['应收账款周转率TTM', '存货周转率TTM', '应付账款周转率TTM']
if all(c in merged_df.columns for c in turnover_cols):
    for c in turnover_cols:
        merged_df[c] = pd.to_numeric(merged_df[c], errors='coerce')
    merged_df['TurnoverIndex'] = merged_df[turnover_cols].mean(axis=1)

if '资产总计' in merged_df:
    merged_df['Size'] = np.log(merged_df['资产总计'].clip(1e-6))
if all(x in merged_df.columns for x in ['负债合计','资产总计']):
    merged_df['Lev'] = merged_df['负债合计'] / merged_df['资产总计']
if all(x in merged_df.columns for x in ['固定资产净额_元','资产总计']):
    merged_df['FixRatio'] = merged_df['固定资产净额_元'] / merged_df['资产总计']
if '员工人数_个' in merged_df:
    merged_df['EmployeeSize'] = np.log(merged_df['员工人数_个'].clip(1))
if all(x in merged_df.columns for x in ['研发投入金额','营业总收入']):
    merged_df['RDIntensity'] = merged_df['研发投入金额'] / merged_df['营业总收入']

# ============================================================
# 8. 1%缩尾生成_w变量
# ============================================================
winsor_cols = [
    'ROA', 'OperatingMargin', 'NetMargin', 'CostRatio', 'CostEfficiency', 'TurnoverIndex',
    'TFP_OP', 'TFP_LP', 'TFP_OLS', 'TFP_FE', 'TFP_GMM',
    'DataEmbed', 'DataEmbed_equal', 'DataEmbed_entropy',
    'Size', 'Lev', 'FixRatio', 'EmployeeSize', 'RDIntensity',
    'ln_GDPpc', 'ln_FiscalTech', 'ln_DigitalInfra'
]
for col in winsor_cols:
    if col in merged_df.columns:
        merged_df[f'{col}_w'] = winsorize(merged_df[col])

# ============================================================
# 9. 滞后项
# ============================================================
merged_df = merged_df.sort_values(['code6', 'year']).reset_index(drop=True)
for var in ['DataEmbed_w', 'DataEmbed_equal_w', 'DataEmbed_entropy_w']:
    if var in merged_df.columns:
        merged_df[f'L1_{var}'] = merged_df.groupby('code6')[var].shift(1)

# ============================================================
# 10. 异质性分组虚拟变量
# ============================================================
if 'evidence_level' in merged_df.columns:
    merged_df['HighEvidence'] = np.where(merged_df['evidence_level'] >= 3, 1, 0)
if 'Size_w' in merged_df.columns:
    merged_df['HighSize'] = np.where(merged_df['Size_w'] > merged_df['Size_w'].median(), 1, 0)
if 'RDIntensity_w' in merged_df.columns:
    merged_df['HighRD'] = np.where(merged_df['RDIntensity_w'] > merged_df['RDIntensity_w'].median(), 1, 0)
if 'ln_DigitalInfra_w' in merged_df.columns:
    merged_df['HighDigitalInfra'] = np.where(merged_df['ln_DigitalInfra_w'] > merged_df['ln_DigitalInfra_w'].median(), 1, 0)

# ============================================================
# 11. 企业内线性插值填充缺失值（仅内插，不向外推）
# ============================================================
interp_cols = [c for c in merged_df.columns if c.endswith('_w')]
merged_df = merged_df.sort_values(['code6', 'year']).reset_index(drop=True)
for col in interp_cols:
    merged_df[col] = merged_df.groupby('code6')[col].transform(
        lambda x: x.interpolate(method='linear', limit_area='inside')
    )
    # 对于仍缺失的首尾记录，保留NaN，避免凭空造数

# ============================================================
# 12. 输出面板与统计
# ============================================================
merged_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\n✅ 面板生成完成，输出文件：{OUTPUT_CSV}")
print(f"样本行数：{merged_df.shape[0]}  列数：{merged_df.shape[1]}")
print(f"企业数量：{merged_df['code6'].nunique()}")
print(f"年份区间：{merged_df['year'].min()} - {merged_df['year'].max()}")
print(f"匹配城市样本数：{merged_df['city'].notna().sum()}")

core_vars = ['code6', 'year', 'city', 'DataEmbed_w', 'ROA_w', 'CostRatio_w', 'TFP_OP_w', 'Size_w', 'Lev_w', 'RDIntensity_w']
print("\n核心变量校验（插值后）：")
for v in core_vars:
    if v in merged_df:
        cnt = merged_df[v].notna().sum()
        print(f"  {v}: 非空{cnt}条 (缺失{792-cnt}条)")
    else:
        print(f"  {v}: 变量不存在")

# 额外输出城市宏观变量的缺失情况
macro_vars = ['ln_GDP_w', 'ln_GDPpc_w', 'ln_FiscalTech_w', 'ln_DigitalInfra_w']
print("\n城市宏观变量缺失统计：")
for v in macro_vars:
    if v in merged_df:
        miss = merged_df[v].isna().sum()
        print(f"  {v}: 缺失{miss}条")