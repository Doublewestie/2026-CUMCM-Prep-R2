"""pytest 共享配置：Code 根目录入 sys.path + 点号文件名模块加载器."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_step_module(filename: str):
    """加载平铺根目录的点号命名模块（如 step1.0_baseline_schedule.py）。"""
    name = filename.split(".")[0].replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
