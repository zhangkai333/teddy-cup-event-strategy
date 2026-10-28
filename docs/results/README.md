# 公开结果说明

本目录只保存可公开的聚合指标，不包含原始行情、公告、个股交易明细或训练模型。

| 文件 | 用途 | 本地来源 |
| --- | --- | --- |
| `ablation_metrics.csv` | E2/E3/E4 消融实验的 F1、Rank IC 与方向准确率 | `runtime/训练结果/5.3_事件影响预测/消融实验结果.csv` |
| `backtest_summary.csv` | 三个历史区间的收益、风险与胜率指标 | `runtime/训练结果/5.4_投资策略构建_扩展/多期回测对比.csv` |
| `classification_metrics.csv` | 事件一级分类指标与评估口径 | `runtime/训练结果/5.1_事件识别与分类/stage2_metadata.json` |

本地存在完整训练结果时，`tests/test_public_results.py` 会检查公开 CSV 是否与原始结果一致；公开仓库的 CI 则校验 CSV、README 和结果图是否相互一致。
