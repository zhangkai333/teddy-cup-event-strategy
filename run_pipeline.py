from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
PYTHON = RUNTIME / ".venv" / "Scripts" / "python.exe"


def run_step(name: str, script: str, args: list[str]) -> None:
    print(f"\n== {name} ==", flush=True)
    command = [str(PYTHON), str(RUNTIME / script), *args]
    result = subprocess.run(command, cwd=RUNTIME, env=os.environ.copy())
    if result.returncode != 0:
        raise SystemExit(f"{name} failed with exit code {result.returncode}.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete local pipeline.")
    parser.add_argument(
        "--skip-historical",
        action="store_true",
        help="Skip the offline historical smoke test.",
    )
    args = parser.parse_args()

    if not PYTHON.exists():
        return 1

    run_step("Train and backtest", "模型训练与回测.py", [])
    run_step("Extended backtest", "拓展3个月回测.py", [])
    run_step("Generate charts", "可视化生成.py", ["--all"])
    if not args.skip_historical:
        run_step(
            "Historical pipeline smoke test",
            "实测.py",
            ["--week", "1", "--use-historical"],
        )

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
