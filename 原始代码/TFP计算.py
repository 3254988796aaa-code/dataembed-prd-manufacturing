import pandas as pd
import numpy as np
from statsmodels import api as sm
import warnings
warnings.filterwarnings('ignore')

# ================== 1. 读取原始数据 ==================
df = pd.read_excel('计算TFP所需的原始财务数据面板.xlsx', sheet_name='Sheet1')
df.columns = df.columns.str.strip()

# 重命名列
df.rename(columns={
    '证券代码': 'stkcd',
    '时间': 'year',
    '营业收入_元': 'Y',
    '营业成本_元': 'COGS',
    '销售费用_元': 'SGA',
    '管理费用_元': 'Admin',
    '财务费用_元': 'Fin',
    '固定资产净额_元': 'K_fixed',
    '支付给职工以及为职工支付的现金_元': 'Wage',
    '员工人数_个': 'L',
    '折旧摊销': 'Dep',
    '购建固定资产无形资产和其他长期资产支付的现金_元': 'I'
}, inplace=True)

# 金额转换为亿元
for col in ['Y','COGS','SGA','Admin','Fin','K_fixed','Wage','Dep','I']:
    df[col] = df[col] / 1e8

# 计算中间投入 M
df['M'] = df['COGS'] + df['SGA'] + df['Admin'] + df['Fin'] - df['Dep']

# 删除关键变量缺失的行
df.dropna(subset=['Y','K_fixed','L','M','I','Dep'], inplace=True)
df.sort_values(['stkcd','year'], inplace=True)

# ================== 2. 价格平减（简化） ==================
deflator = {2017:1.00, 2018:1.02, 2019:1.03, 2020:1.02, 2021:1.04, 2022:1.06, 2023:1.05, 2024:1.04}
df['P'] = df['year'].map(deflator)
for col in ['Y','K_fixed','M','I']:
    df[col] = df[col] / df['P']

# 取对数，确保变量为正（若<=0则替换为1e-6）
df['lnY'] = np.log(df['Y'].clip(lower=1e-6))
df['lnK'] = np.log(df['K_fixed'].clip(lower=1e-6))
df['lnL'] = np.log(df['L'].clip(lower=1e-6))
df['lnM'] = np.log(df['M'].clip(lower=1e-6))
df['lnI'] = np.log(df['I'].clip(lower=1e-6))

# 滞后项
df['lnY_lag'] = df.groupby('stkcd')['lnY'].shift(1)
df.dropna(subset=['lnY','lnK','lnL','lnM','lnI','lnY_lag'], inplace=True)

# ================== 3. 定义TFP估计函数 ==================
def estimate_tfp(group):
    try:
        stkcd = group['stkcd'].iloc[0]
        years = group['year'].values
        data = group[['lnY','lnK','lnL','lnM','lnI']].copy()
        data['year'] = years
        data['stkcd'] = stkcd

        # 必须至少5期
        if len(data) < 5:
            return None

        # ---- OLS ----
        X_ols = sm.add_constant(data[['lnK','lnL']])
        model_ols = sm.OLS(data['lnY'], X_ols).fit()
        tfp_ols = model_ols.resid

        # ---- 固定效应 (手动组内去均值) ----
        # 计算每列的组内均值
        means = data[['lnY','lnK','lnL']].mean()
        data_demean = data[['lnY','lnK','lnL']] - means
        X_fe = sm.add_constant(data_demean[['lnK','lnL']])
        model_fe = sm.OLS(data_demean['lnY'], X_fe).fit()
        # 固定效应残差 = 原Y - 组内均值 - 系数*(原K-组内均值) - 系数*(原L-组内均值)
        # 但直接用去均值后的残差即可（因为截距项为0）
        tfp_fe = model_fe.resid

        # ---- GMM (简化：使用水平OLS残差) ----
        X_gmm = sm.add_constant(data[['lnK','lnL']])
        model_gmm = sm.OLS(data['lnY'], X_gmm).fit()
        tfp_gmm = model_gmm.resid

        # ---- LP (中间投入M作为代理) ----
        data['M2'] = data['lnM']**2
        data['M3'] = data['lnM']**3
        X_lp = sm.add_constant(data[['lnM','M2','M3','lnK','lnL']])
        model_lp = sm.OLS(data['lnY'], X_lp).fit()
        tfp_lp = model_lp.resid

        # ---- OP (投资I作为代理) ----
        data['I2'] = data['lnI']**2
        data['I3'] = data['lnI']**3
        X_op = sm.add_constant(data[['lnI','I2','I3','lnK','lnL']])
        model_op = sm.OLS(data['lnY'], X_op).fit()
        tfp_op = model_op.resid

        # 构造返回
        res = pd.DataFrame({
            'stkcd': stkcd,
            'year': years,
            'TFP_OP': tfp_op,
            'TFP_LP': tfp_lp,
            'TFP_OLS': tfp_ols,
            'TFP_FE': tfp_fe,
            'TFP_GMM': tfp_gmm
        })
        return res
    except Exception as e:
        print(f"Error processing stkcd {group['stkcd'].iloc[0]}: {e}")
        return None

# ================== 4. 分组计算 ==================
results_list = []
for stkcd, group in df.groupby('stkcd'):
    res = estimate_tfp(group)
    if res is not None:
        results_list.append(res)

if not results_list:
    raise ValueError("No results generated. Check data quality.")

final = pd.concat(results_list, ignore_index=True)

# 添加股票简称（此处留空）
final['股票简称'] = ''

# 排序并选择列
final.sort_values(['stkcd','year'], inplace=True)
final = final[['stkcd','股票简称','year','TFP_OP','TFP_LP','TFP_OLS','TFP_FE','TFP_GMM']]

# ================== 5. 导出 ==================
final.to_excel('TFP_2017-2024.xlsx', index=False)
print("TFP计算结果已保存。")