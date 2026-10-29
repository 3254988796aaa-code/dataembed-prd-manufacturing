"""本地数据路径与配置；默认使用仓库随附数据。"""
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path(os.environ.get("DATAEMBED_INPUT_DIR", str(ROOT / "输入数据"))).expanduser().resolve()
OUTPUT_DIR = Path(os.environ.get("DATAEMBED_OUTPUT_DIR", str(ROOT / "local_outputs"))).expanduser().resolve()
ANNUAL_REPORT_DIR = Path(os.environ.get("DATAEMBED_ANNUAL_REPORT_DIR", str(ROOT / "原始数据" / "公司年报PDF"))).expanduser().resolve()


def local_input(name, required=True):
    target = INPUT_DIR / name
    if required and not target.is_file():
        raise FileNotFoundError(f"缺少本地输入 {name}。请检查输入数据目录，参阅 docs/本地运行.md。")
    return target


def local_output(name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / name


def load_deflator(filename):
    import pandas as pd
    frame = pd.read_csv(local_input(filename))
    if not {"year", "deflator"}.issubset(frame.columns):
        raise ValueError(f"{filename} 需要 year、deflator 两列。")
    years = pd.to_numeric(frame["year"], errors="raise")
    values = pd.to_numeric(frame["deflator"], errors="raise")
    if years.isna().any() or (years % 1 != 0).any() or years.duplicated().any():
        raise ValueError("平减指数年份必须是唯一的整数。")
    if values.isna().any() or (values <= 0).any() or (~values.map(math.isfinite)).any():
        raise ValueError("平减指数必须是有限正数。")
    return dict(zip(years.astype(int), values.astype(float)))


def load_stock_city_data():
    import pandas as pd
    frame = pd.read_csv(local_input("stock_city_map.csv"), dtype=str)
    columns = ["firm_name", "code", "city"]
    if not set(columns).issubset(frame.columns):
        raise ValueError("stock_city_map.csv 需要 firm_name、code、city 三列。")
    if frame[columns].isna().any().any():
        raise ValueError("企业映射字段不能缺失。")
    return list(frame[columns].itertuples(index=False, name=None))
