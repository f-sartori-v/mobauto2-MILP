import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

yaml_path = PROJECT_DIR / "setups" / "nice30.yaml"
csv_path = SCRIPT_DIR / "nice10_grouped_direction_counts.csv"
png_path = SCRIPT_DIR / "nice10_requests_by_time_and_direction.png"



with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

requests = pd.DataFrame(data["requests"])

summary = (
    requests.groupby(["time", "dir"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
    .sort_values("time")
)

for col in ["OUT", "RET"]:
    if col not in summary.columns:
        summary[col] = 0

base_minutes = 7 * 60
summary["clock_time"] = summary["time"].apply(
    lambda m: f"{(base_minutes + m)//60}:{(base_minutes + m)%60:02d}"
)

summary[["time", "clock_time", "OUT", "RET"]].to_csv(csv_path, index=False)

x = range(len(summary))
width = 0.4

plt.figure(figsize=(14, 6))
plt.bar([i - width/2 for i in x], summary["OUT"], width=width, label="OUT")
plt.bar([i + width/2 for i in x], summary["RET"], width=width, label="RET")

plt.xticks(list(x), summary["clock_time"], rotation=45)
plt.xlabel("Time")
plt.ylabel("Number of requests")
plt.title("Requests by time slot and direction (nice10)")
plt.legend()
plt.tight_layout()

plt.savefig(png_path, dpi=200, bbox_inches="tight")
plt.show()

summary[["clock_time", "OUT", "RET"]]