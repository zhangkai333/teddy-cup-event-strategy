from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from runtime.metrics import parse_percentage


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"


def test_public_ablation_results_match_documented_values() -> None:
    results = pd.read_csv(RESULTS / "ablation_metrics.csv", encoding="utf-8-sig")
    results["实验"] = results["实验"].astype(str)

    expected = {
        "E2": (0.4210, 0.0908, 0.5659),
        "E3": (0.4229, 0.1478, 0.5691),
        "E4": (0.4286, 0.1747, 0.5734),
    }
    for model, (f1, rank_ic, direction_accuracy) in expected.items():
        row = results.loc[results["实验"].str.startswith(model)]
        assert len(row) == 1
        assert row.iloc[0]["分类F1"] == pytest.approx(f1)
        assert row.iloc[0]["回归Rank_IC"] == pytest.approx(rank_ic)
        assert row.iloc[0]["方向准确率"] == pytest.approx(direction_accuracy)


def test_visualization_reads_ablation_values_instead_of_hardcoding() -> None:
    script = ROOT / "runtime" / "可视化生成.py"
    spec = importlib.util.spec_from_file_location("teddy_visualization", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    results = module.load_ablation_metrics().set_index("code")

    assert results.loc["E2", "rank_ic"] == pytest.approx(0.0908)
    assert results.loc["E3", "rank_ic"] == pytest.approx(0.1478)
    assert results.loc["E4", "rank_ic"] == pytest.approx(0.1747)


def test_public_backtest_results_match_readme() -> None:
    results = pd.read_csv(
        RESULTS / "backtest_summary.csv",
        encoding="utf-8-sig",
        dtype=str,
    ).set_index("区间")

    assert results.loc["3周", "策略收益"] == "5.66%"
    assert results.loc["3周", "Sharpe"] == "2.985"
    assert results.loc["3个月", "策略收益"] == "5.09%"
    assert results.loc["3个月", "最大回撤"] == "-7.81%"
    assert results.loc["单月", "超额收益"] == "-0.20%"
    assert parse_percentage(results.loc["3个月", "策略收益"]) == pytest.approx(0.0509)


def test_public_classification_metrics_round_to_readme_values() -> None:
    results = pd.read_csv(
        RESULTS / "classification_metrics.csv",
        encoding="utf-8-sig",
    ).set_index("指标")

    weighted_f1 = float(results.loc["Weighted F1", "数值"])
    accuracy = float(results.loc["Accuracy", "数值"])
    assert weighted_f1 == pytest.approx(0.97107)
    assert accuracy == pytest.approx(0.96830)
    assert f"{weighted_f1:.2%}" == "97.11%"


def test_public_assets_and_readme_links_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for relative_path in [
        "docs/results/ablation_metrics.csv",
        "docs/results/backtest_summary.csv",
        "docs/assets/02_ablation_rank_ic.png",
        "docs/assets/03_backtest_equity_curve.png",
    ]:
        assert (ROOT / relative_path).is_file()
        assert relative_path in readme


def test_readme_keeps_core_metrics_consistent() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for expected_text in [
        "0.1747",
        "0.0908",
        "0.1478",
        "5.09%",
        "-7.81%",
        "97.11%",
        "随机 80/20 验证集",
    ]:
        assert expected_text in readme


def test_local_raw_results_match_public_exports_when_available() -> None:
    raw_ablation = (
        ROOT
        / "runtime"
        / "训练结果"
        / "5.3_事件影响预测"
        / "消融实验结果.csv"
    )
    if not raw_ablation.exists():
        pytest.skip("Raw training results are intentionally excluded from the repository.")

    raw = pd.read_csv(raw_ablation, encoding="utf-8-sig")
    public = pd.read_csv(RESULTS / "ablation_metrics.csv", encoding="utf-8-sig")
    raw["实验"] = raw["实验"].astype(str)
    public["实验"] = public["实验"].astype(str)

    for model in ["E2", "E3", "E4"]:
        raw_row = raw.loc[raw["实验"].str.startswith(model)].iloc[0]
        public_row = public.loc[public["实验"].str.startswith(model)].iloc[0]
        assert public_row["分类F1"] == pytest.approx(
            pd.to_numeric(raw_row["分类F1"], errors="coerce")
        )
        assert public_row["回归Rank_IC"] == pytest.approx(
            pd.to_numeric(raw_row["回归Rank_IC"], errors="coerce")
        )
        assert public_row["方向准确率"] == pytest.approx(
            pd.to_numeric(raw_row["方向准确率"], errors="coerce")
        )

    raw_backtest = (
        ROOT
        / "runtime"
        / "训练结果"
        / "5.4_投资策略构建_扩展"
        / "多期回测对比.csv"
    )
    if not raw_backtest.exists():
        pytest.skip("Raw backtest summary is intentionally excluded from the repository.")

    raw_periods = pd.read_csv(raw_backtest, encoding="utf-8-sig").set_index("回测期")
    public_periods = pd.read_csv(
        RESULTS / "backtest_summary.csv",
        encoding="utf-8-sig",
    ).set_index("区间")
    period_map = {
        "3周": "3周(赛题期)",
        "单月": "12月整月",
        "3个月": "3个月(10-12月)",
    }
    percentage_columns = {
        "策略收益": "累计收益率",
        "沪深300": "沪深300基准",
        "超额收益": "超额收益",
        "最大回撤": "最大回撤",
        "周胜率": "周胜率",
    }
    for public_period, raw_period in period_map.items():
        for public_col, raw_col in percentage_columns.items():
            assert parse_percentage(public_periods.loc[public_period, public_col]) == pytest.approx(
                parse_percentage(raw_periods.loc[raw_period, raw_col])
            )
        assert float(public_periods.loc[public_period, "Sharpe"]) == pytest.approx(
            float(raw_periods.loc[raw_period, "夏普比"])
        )

    raw_classification = (
        ROOT
        / "runtime"
        / "训练结果"
        / "5.1_事件识别与分类"
        / "stage2_metadata.json"
    )
    if raw_classification.exists():
        metadata = json.loads(raw_classification.read_text(encoding="utf-8"))
        public_classification = pd.read_csv(
            RESULTS / "classification_metrics.csv",
            encoding="utf-8-sig",
        ).set_index("指标")
        assert float(public_classification.loc["Weighted F1", "数值"]) == pytest.approx(
            metadata["c1_f1"], abs=1e-5
        )
        assert float(public_classification.loc["Accuracy", "数值"]) == pytest.approx(
            metadata["c1_acc"], abs=1e-5
        )
