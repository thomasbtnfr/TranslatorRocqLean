import json
import re
import subprocess
import tempfile
from pathlib import Path

from trl_llm.train.config import TrainingConfig


def load_visual_prompt(path: Path) -> list:
    """Load the 'data' field from the visual prompt JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)["data"]

def extract_step(text: str) -> int:
    """Extract step number from a string like 'Step 123:'."""
    if match := re.search(r"Step (\d+):", text):
        return int(match.group(1))
    raise ValueError(f"Step number not found in: {text}")


def extract_code(text: str) -> str | bool:
    """Extract Rocq/Lean code block enclosed in ```rocq ... ```."""
    if match := re.search(r"```(?:rocq|coq|lean)\n(.*?)```", text, re.DOTALL):
        return match.group(1).strip()
    return False

def check_rocq_code(code: str) -> tuple[bool, str | None]:
    """Write Rocq code to a temp file and run coqc on it."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".v", delete=True) as tmp_file:
        tmp_file.write(code)
        tmp_file.flush()
        try:
            subprocess.run(["coqc", tmp_file.name],
                                    check=True,
                                    capture_output=True,
                                    text=True
                                    )
            return True, None
        except subprocess.CalledProcessError as e:
            tmp_file.seek(0)
            return False, e.stderr

def check_lean_code(code: str, config: TrainingConfig) -> tuple[bool, str | None]:
    with open(config.compile_lean_full_file_path, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        subprocess.run(["lake", "env", "lean", config.compile_lean_full_file_path],
                                check=True,
                                capture_output=True,
                                text=True
        )
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stdout