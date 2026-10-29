# DataEmbed｜珠三角装备制造企业数据要素研究

从企业年报文本出发，构建数据要素嵌入度指标，连接面板回归、企业阶段诊断和交互式决策分析。

个人独立完成的数据分析项目。作者：[3254988796aaa-code](https://github.com/3254988796aaa-code)。

本仓库收录完整项目：原始资料、输入数据、分析代码、既有结果图表与研究报告。原项目共 876 个文件、约 3.7 GiB；逐文件校验信息见 [项目文件校验.csv](项目文件校验.csv)。

## 项目内容

| 环节 | 实现 |
| --- | --- |
| 年报文本处理 | PyMuPDF 提取文本、关键词识别、证据句筛选与等级判定 |
| 指标构建 | PCA、等权和熵权三种 DataEmbed 指标 |
| 数据工程 | Pandas / Polars 面板合并、证券代码规范化、缺失值处理与缩尾 |
| 实证分析 | 双向去均值、企业聚类标准误、滞后与机制检验、异质性及安慰剂检验 |
| 应用分析 | 企业阶段诊断、规则建议、策略反事实模拟及样本外回测 |
| 交互展示 | Streamlit + Plotly 仪表盘 |

## 完整下载

PDF、Excel、压缩包、报告和图片通过 Git LFS 存储。请先安装 Git LFS，再下载仓库：

```bash
git lfs install
git clone https://github.com/3254988796aaa-code/dataembed-prd-manufacturing.git
cd dataembed-prd-manufacturing
git lfs pull
```

## 目录

| 目录或文件 | 内容 |
| --- | --- |
| [代码/](代码/) | 便于运行的整理版，支持仓库相对路径与本地输出目录 |
| [原始代码/](原始代码/) | 原项目 10 个脚本原件，保留原始配置及内嵌数值 |
| [原始数据/](原始数据/) | 776 份企业报告 PDF、财务及 TFP 压缩包、资料说明 |
| [输入数据/](输入数据/) | 面板、财务、城市宏观、政策名单及指标构建输入 |
| [运行结果/](运行结果/) | 原研究 CSV、Excel、日志、结果图和策略回测输出 |
| [研究报告](<数据要素的阶段性赋能效应研究—— 基于广东珠三角装备制造业企业实证与转型模拟.doc>) | 完整研究报告原件 |
| [年报索引](data/annual_reports/annual_report_index.csv) | 文件名、年份、类型与大小 |
| [docs/](docs/) | 运行、资料范围与方法说明 |

## 运行

建议使用 Python 3.10 或更高版本。在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python 代码/run_all.py
python -m streamlit run 代码/app.py
```

整理版默认从 输入数据/ 读取，新增分析结果写入 local_outputs/，原有 运行结果/ 保留作对照。年报文本处理默认使用 原始数据/公司年报PDF/，首次全文处理耗时较长，可先查看帮助并用 --max_pages 检查流程。

- [完整运行说明](docs/本地运行.md)
- [项目资料范围](docs/数据发布范围.md)
- [方法与当前限制](docs/方法与限制.md)

提交前检查可以验证文件范围、大文件指针及原项目文件的完整性：

```bash
git config core.hooksPath .githooks
python tools/check_public_files.py
```
