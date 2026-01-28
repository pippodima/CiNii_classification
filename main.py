from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n==> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)

def main() -> int:
    try:
        # 1) run shell stage
        # sh = ROOT / "data/scripts/sample_resourcedumps.sh"
        # run(["bash", str(sh)])  # avoids executable-bit issues

        # 2) run python stages
        run([sys.executable, str(ROOT / "data/scripts/parse_nii_documents.py"), "-max-document", "30000"])
        run([sys.executable, str(ROOT / "data/scripts/process_data.py")])
        run([sys.executable, str(ROOT / "data/scripts/main.py")])

        print("\n✅ Pipeline finished")
        return
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Step failed with exit code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    raise SystemExit(main())
