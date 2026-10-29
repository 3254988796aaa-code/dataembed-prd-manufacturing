"""检查 Git 暂存区的公开文件范围，不读取未暂存的本地数据。"""
import ast
import csv
import io
import re
import subprocess
import sys

ALLOWED = {
    ".gitignore", ".gitattributes", "README.md", "requirements.txt",
    ".githooks/pre-commit", "tools/check_public_files.py",
    "docs/数据发布范围.md", "docs/本地运行.md", "docs/方法与限制.md",
    "data/annual_reports/README.md", "data/annual_reports/annual_report_index.csv",
    "代码/_local_data.py", "代码/app.py", "代码/make_alias.py", "代码/run_all.py",
    "代码/strategy_backtest.py", "代码/TFP计算.py", "代码/年报处理.py",
    "代码/年报文本处理（Numba JIT 加速版）.py", "代码/整合生成最终面板数据.py",
    "代码/整合面板数据（Polars 全兼容版）.py",
}


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    names = [p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p]
    errors = []
    for name in names:
        if name not in ALLOWED:
            errors.append(f"不在公开允许清单：{name}")
            continue
        content = git("show", ":" + name)
        entry = git("ls-files", "--stage", "--", name)
        if not entry.startswith(b"100644 ") and not entry.startswith(b"100755 "):
            errors.append(f"不允许链接或特殊文件：{name}")
            continue
        if len(content) > 1024 * 1024:
            errors.append(f"公开文本文件过大：{name}")
            continue
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            errors.append(f"仅允许 UTF-8 文本文件：{name}")
            continue
        if re.search(r"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|" + r"-----BEGIN " + r"[A-Z ]*PRIVATE KEY-----", text):
            errors.append(f"疑似凭据：{name}")
        if name.endswith(".py"):
            try:
                tree = ast.parse(text, filename=name)
            except SyntaxError:
                errors.append(f"Python 语法错误：{name}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
                    if targets & {"PPI_DEFLATOR", "deflator", "STOCK_CITY_DATA", "base_roa_coef", "base_cost_coef"}:
                        if isinstance(node.value, (ast.Constant, ast.Dict, ast.List, ast.Tuple, ast.UnaryOp)):
                            errors.append(f"存在内嵌经济配置或结果：{name}:{node.lineno}")
        if name.endswith("annual_report_index.csv"):
            reader = csv.DictReader(io.StringIO(text))
            if reader.fieldnames != ["filename", "year", "report_type", "size_bytes"]:
                errors.append("年报索引字段范围不符")
            else:
                for row in reader:
                    if not row["filename"].lower().endswith(".pdf") or not row["size_bytes"].isdigit():
                        errors.append("年报索引包含异常条目")
                        break
    if errors:
        print("\n".join(errors))
        return 1
    print(f"公开文件检查通过：{len(names)} 个暂存文件；文件范围、文本格式、语法及已知内嵌数据检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
