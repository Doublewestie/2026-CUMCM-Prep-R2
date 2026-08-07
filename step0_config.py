"""step0_config — 全局路径与常量配置（全题共享）."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "output"
FIGURES = ROOT / "figures"
CLEAN = OUTPUT / "clean"

REGIONS = ["RegionA", "RegionB", "RegionC", "RegionD", "RegionE", "RegionF"]
TASK_TYPES = ["RealTimeInference", "BatchInference", "AITraining"]
TASK_TYPE_SHORT = {"RealTimeInference": "RT", "BatchInference": "BI", "AITraining": "AT"}
MAX_LATENCY = {"RealTimeInference": 20, "BatchInference": 80, "AITraining": 150}
GPU_POWER_MW = {"RealTimeInference": 0.08, "BatchInference": 0.10, "AITraining": 0.16}
PRIORITY = {"RealTimeInference": 3, "BatchInference": 2, "AITraining": 1}

MAIN_END = 2399          # 主时域末（任务到达上界）
CLOSURE_END = 2405       # 收尾时域末
SETTLE_HOUR = 2406       # 终态结算时点
HOURS_MAIN = 2400
HOURS_TOTAL = 2407       # 0..2406

FILES = {
    "workload": "workload_trace.xlsx",
    "region_time": "region_time_data.xlsx",
    "gpu": "GPU_information.xlsx",
    "latency": "network_latency.xlsx",
    "power": "power_mapping.xlsx",
    "storage": "storage_information.xlsx",
}
