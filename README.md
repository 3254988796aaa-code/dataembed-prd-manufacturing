# DataEmbed｜珠三角装备制造企业数据要素研究

从企业年报文本出发，构建数据要素嵌入度指标，连接面板回归、企业阶段诊断和交互式决策分析。

个人独立完成的数据分析项目。作者：[3254988796aaa-code](https://github.com/3254988796aaa-code)。

> 本仓库为代码公开版。经济数据库、财务及城市数据、处理后面板、研究结果、图表和完整报告均不发布。年报资料提供文件索引，PDF 原件保留在本地。

## 项目内容

| 环节 | 实现 |
| --- | --- |
| 年报文本处理 | PyMuPDF 提取文本、关键词识别、证据句筛选与等级判定 |
| 指标构建 | PCA、等权和熵权三种 DataEmbed 指标 |
| 数据工程 | Pandas / Polars 面板合并、证券代码规范化、缺失值处理与缩尾 |
| 实证分析 | 双向去均值、企业聚类标准误、滞后与机制检验、异质性及安慰剂检验 |
| 应用分析 | 企业阶段诊断、规则建议、策略反事实模拟及样本外回测 |
| 交互展示 | Streamlit + Plotly 仪表盘 |

## 目录

```text
代码/                    分析脚本与本地数据配置
  年报处理.py             年报文本处理
  年报文本处理（Numba JIT 加速版）.py
  整合生成最终面板数据.py
  整合面板数据（Polars 全兼容版）.py
  TFP计算.py
  run_all.py             主分析流程
  strategy_backtest.py   策略模拟及回测
  app.py                 交互仪表盘
  make_alias.py          本地企业别名映射生成
  _local_data.py         数据路径与私有配置读取
data/annual_reports/     年报文件索引及说明
docs/                    本地运行、公开范围与方法限制
tools/                   发布文件检查
.githooks/               本地提交检查
```

## 使用

建议使用 Python 3.10 或更高版本，在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
python 代码/年报处理.py --help
python 代码/run_all.py --help
python -m streamlit run 代码/app.py
```

仪表盘在未配置数据时会提示准备本地输入。完整研究流程需要使用者自行提供有使用权限的本地数据，具体文件和字段见 [本地运行](docs/本地运行.md)。

- [年报资料说明](data/annual_reports/README.md)
- [数据发布范围](docs/数据发布范围.md)
- [方法与当前限制](docs/方法与限制.md)

首次使用本地仓库时启用提交检查：

```bash
git config core.hooksPath .githooks
python tools/check_public_files.py
```

当前代码与文档保留项目研究方法，公开仓库不提供原始经济数据或实证结果复现包。
