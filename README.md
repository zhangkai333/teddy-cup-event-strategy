# 泰迪杯事件驱动型股市投资策略

本项目基于泰迪杯完整数据集完成事件分类、关联公司挖掘、事件影响预测、组合回测和可视化分析。

## 数据约定

仓库不提交原始 CSV、虚拟环境和生成结果。运行时通过环境变量
`TEDDY_DATA_DIR` 指定本地完整数据集目录，目录下应包含：

```text
公司财务数据/
股吧评论数据/
股票行情数据/
建模核心表/
事件数据/
训练结果/          # 可选，部分基线结果会从此目录读取
```

未设置 `TEDDY_DATA_DIR` 时，脚本默认读取 `runtime/数据集_清洗`。该目录已加入
`.gitignore`，不会上传 GitHub。

## 环境准备

Python 3.12 已验证可用。

```powershell
cd "C:\path\to\泰迪杯熟悉项目"
py -3.12 -m venv runtime\.venv
runtime\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 一键运行

将 `DataDir` 替换为本地泰迪杯完整数据集的实际路径：

```powershell
$env:TEDDY_DATA_DIR = "C:\path\to\使用数据集"
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -SkipInstall
```

脚本依次执行：

1. 模型训练与三周回测
2. 三个月扩展回测
3. 35 张 EDA、回测和消融实验图表
4. 历史测试集实测冒烟测试

任一步骤失败时脚本会立即返回非零退出码。

## 单独运行

```powershell
$env:TEDDY_DATA_DIR = "C:\path\to\使用数据集"
cd runtime

.\.venv\Scripts\python.exe .\模型训练与回测.py
.\.venv\Scripts\python.exe .\拓展3个月回测.py
.\.venv\Scripts\python.exe .\可视化生成.py --all
.\.venv\Scripts\python.exe .\实测.py --week 1 --use-historical
```

`实测.py --use-historical` 仅用于离线联调。正式实测默认使用实时网络数据，
部分数据源可能需要额外安装 `akshare`、`baostock` 或配置 token。

## 输出目录

- `runtime/训练结果`：模型、预测结果和回测交易记录
- `runtime/实测结果`：历史实测结果与 `result.xlsx`
- `runtime/结果（图表，分析）`：论文和汇报用图表

以上目录均已加入 `.gitignore`，需要提交示例结果时可单独调整规则。
