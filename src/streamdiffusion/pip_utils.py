import importlib
import importlib.util
import os
import subprocess
import sys
from typing import Dict, Optional

from packaging.version import Version


python = sys.executable
index_url = os.environ.get("INDEX_URL", "")


def version(package: str) -> Optional[Version]:
    try:
        return Version(importlib.import_module(package).__version__)
    except ModuleNotFoundError:
        return None


def is_installed(package: str) -> bool:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


def run_python(command: str, env: Dict[str, str] = None) -> str:
    run_kwargs = {
        "args": f"\"{python}\" {command}",
        "shell": True,
        "env": os.environ if env is None else env,
        "encoding": "utf8",
        "errors": "ignore",
    }

    print(run_kwargs["args"])

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        print(f"Error running command: {command}", file=sys.stderr)
        raise RuntimeError(f"Error running command: {command}")

    return result.stdout or ""


def run_pip(command: str, env: Dict[str, str] = None) -> str:
    return run_python(f"-m pip {command}", env)
