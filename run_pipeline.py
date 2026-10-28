from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"


def resolve_python(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Python interpreter does not exist: {candidate}")
        return candidate

    for candidate in [
        RUNTIME / ".venv" / "Scripts" / "python.exe",
        RUNTIME / ".venv" / "bin" / "python",
    ]:
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def resolve_data_dir(explicit: str | None) -> Path:
    raw_value = explicit or os.environ.get("TEDDY_DATA_DIR")
    data_dir = (
        Path(raw_value).expanduser().resolve()
        if raw_value
        else RUNTIME / "数据集_清洗"
    )
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}. "
            "Pass --data-dir or set TEDDY_DATA_DIR."
        )
    return data_dir


def run_step(
    name: str,
    script: str,
    args: list[str],
    python: Path,
    env: dict[str, str],
) -> None:
    print(f"\n== {name} ==", flush=True)
    command = [str(python), str(RUNTIME / script), *args]
    result = subprocess.run(command, cwd=RUNTIME, env=env)
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete local pipeline.")
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip the offline historical smoke test.",
    )
    parser.add_argument(
        "--data-dir",
        help="Directory containing the cleaned datasets. Defaults to TEDDY_DATA_DIR.",
    )
    parser.add_argument(
        "--python",
        help="Python interpreter to use. Defaults to runtime/.venv when available.",
    )
    args = parser.parse_args()

    try:
        python = resolve_python(args.python)
        data_dir = resolve_data_dir(args.data_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    env = os.environ.copy()
    env["TEDDY_DATA_DIR"] = str(data_dir)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    print(f"Python: {python}")
    print(f"Data:   {data_dir}")

    run_step("Train and backtest", "模型训练与回测.py", [], python, env)
    run_step("Extended backtest", "拓展3个月回测.py", [], python, env)
    run_step("Generate charts", "可视化生成.py", ["--all"], python, env)
    if not args.skip_historical:
        run_step(
            "Historical pipeline smoke test",
            "实测.py",
            ["--week", "1", "--use-historical"],
            python,
            env,
        )

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
