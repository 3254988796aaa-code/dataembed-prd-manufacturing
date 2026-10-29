import pandas as pd
import re
from _local_data import local_input

df = pd.read_excel(local_input("副本已处理_股票代码.xlsx"), dtype=str)


# 补全股票代码为 6 位数字（去掉非数字字符，再 zfill）
def clean_code(code):
    if pd.isna(code):
        return ""
    s = re.sub(r"\D", "", str(code))  # 去除非数字
    return s.zfill(6)  # 补足 6 位


df["stock_code"] = df["股票代码"].apply(clean_code)

# 构造别名：主名称（去除括号及内容） + 可能的简称
rows = []
for _, r in df.iterrows():
    raw_name = r["企业名称"]
    # 提取主名称：去掉括号及其内容、去掉“（”、“）”等
    main_name = re.sub(r"\(.*?\)", "", raw_name)  # 去除英文括号内容
    main_name = re.sub(r"（.*?）", "", main_name)  # 去除中文括号内容
    main_name = main_name.strip()

    # 如果主名称包含“/”等，取第一部分（如“科达制造”）
    main_name = main_name.split("/")[0].strip()

    # 别名列表：主名称本身 + 可能的一些常见简称（如“美的”）
    aliases = [main_name]
    # 针对特殊企业手动添加别名（可根据你 PDF 中实际出现的名称补充）
    if "美的" in main_name:
        aliases.append("美的")
    if "科达制造" in main_name:
        aliases.append("科达洁能")  # 历史名称
    if "南兴装备" in main_name:
        aliases.append("南兴股份")  # 可能改名
    # 更多特殊可自行添加...

    for alias in set(aliases):
        if alias:
            rows.append({
                "stock_code": r["stock_code"],
                "firm_name": alias,
                "aliases": ""  # 留空，也可以填入其他别名
            })

# 去重（同一个 code 可能对应多个别名，保留即可）
alias_df = pd.DataFrame(rows).drop_duplicates()

# 保存为 CSV（UTF-8 编码）
alias_df.to_csv(local_input("alias_map.csv", required=False), index=False, encoding="utf-8-sig")
print(f"生成 alias_map.csv，共 {len(alias_df)} 条映射记录。")