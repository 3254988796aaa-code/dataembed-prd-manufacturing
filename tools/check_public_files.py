"""检查暂存文件、Git LFS 指针及原项目文件完整性。"""
import ast
import csv
import hashlib
import io
import pathlib
import re
import subprocess
import sys

FIXED = {
    '.gitignore', '.gitattributes', 'README.md', 'requirements.txt', '组员说明.txt', '项目文件校验.csv',
    'docs/数据发布范围.md', 'docs/本地运行.md', 'docs/方法与限制.md',
    'tools/check_public_files.py', 'data/annual_reports/README.md', 'data/annual_reports/annual_report_index.csv',
    '.githooks/pre-commit', '.githooks/pre-push', '.githooks/post-checkout', '.githooks/post-commit', '.githooks/post-merge',
}
ASSET_ROOTS = {'原始代码', '原始数据', '输入数据', '运行结果'}
ASSET_SUFFIXES = {'.py', '.pdf', '.zip', '.xlsx', '.xls', '.csv', '.txt', '.png', '.doc', '.docx'}
LFS_SUFFIXES = {'.pdf', '.zip', '.xlsx', '.xls', '.doc', '.docx', '.png'}
POINTER = re.compile(rb'version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize ([0-9]+)\n')


def main():
    raw = subprocess.check_output(['git', 'ls-files', '--stage', '-z'])
    entries = []
    for item in raw.split(b'\0'):
        if item:
            fields, name = item.split(b'\t', 1)
            mode, oid, stage = fields.split()
            entries.append((name.decode('utf-8'), mode, oid, stage))
    result = subprocess.run(['git', 'cat-file', '--batch'], input=b'\n'.join(e[2] for e in entries)+b'\n', capture_output=True, check=True)
    stream = io.BytesIO(result.stdout)
    errors, info = [], {}
    manifest = None
    lfs_count = 0
    secret = re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|' + r'-----BEGIN ' + r'[A-Z ]*PRIVATE KEY-----')
    for name, mode, oid, stage in entries:
        header = stream.readline().split()
        content = stream.read(int(header[2])); stream.read(1)
        parts = pathlib.PurePosixPath(name).parts
        suffix = pathlib.PurePosixPath(name).suffix.lower()
        allowed = name in FIXED or (parts[0] in ASSET_ROOTS and len(parts)>1 and suffix in ASSET_SUFFIXES) or (parts[0]=='代码' and len(parts)==2 and suffix=='.py') or (len(parts)==1 and suffix in {'.doc','.docx'})
        if not allowed:
            errors.append(f'未识别的项目文件：{name}')
        if mode not in {b'100644',b'100755'} or stage!=b'0':
            errors.append(f'不允许链接、特殊文件或未解决的冲突：{name}')
        if suffix in LFS_SUFFIXES:
            pointer = POINTER.fullmatch(content)
            if not pointer:
                errors.append(f'二进制资料必须通过 Git LFS 存储：{name}')
                continue
            digest, size = pointer.groups()
            info[name] = (digest.decode(), int(size)); lfs_count += 1
            continue
        info[name] = (hashlib.sha256(content).hexdigest(),len(content))
        if len(content)>100*1024*1024:
            errors.append(f'普通 Git 文件过大：{name}')
            continue
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError:
            try: text = content.decode('gb18030')
            except UnicodeDecodeError:
                errors.append(f'无法识别文本编码：{name}')
                continue
        if secret.search(text):
            errors.append(f'疑似凭据：{name}')
        if suffix=='.py':
            try: ast.parse(text,filename=name)
            except SyntaxError: errors.append(f'Python 语法错误：{name}')
        if name=='项目文件校验.csv': manifest=text
    originals=0
    if manifest is None:
        errors.append('缺少原项目文件校验清单')
    else:
        seen=set()
        for row in csv.DictReader(io.StringIO(manifest)):
            name=row['repository_path']; originals+=1
            if name in seen: errors.append(f'原件清单重复：{name}')
            seen.add(name)
            if info.get(name)!=(row['sha256'],int(row['size_bytes'])):
                errors.append(f'原项目文件缺失或内容不符：{name}')
    if errors:
        print('\n'.join(errors));return 1
    print(f'项目检查通过：{len(entries)} 个文件，{lfs_count} 个 LFS 文件，{originals} 个原项目文件校验一致。')
    return 0


if __name__=='__main__':
    sys.exit(main())
