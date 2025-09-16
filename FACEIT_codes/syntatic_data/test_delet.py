import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# STEP 1 — Base directory and output folder
parent_dir = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change"
summary_dir = os.path.join(parent_dir, "summary_plots")
os.makedirs(summary_dir, exist_ok=True)

# STEP 2 — Load all Excel results
excel_paths = glob.glob(os.path.join(parent_dir, "series*", "result2", "pupil_detection_results.xlsx"))
print(f"Found {len(excel_paths)} result files.")

# STEP 3 — Load and label
all_summaries = []
for path in excel_paths:
    video_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    df = pd.read_excel(path, sheet_name="Summary")
    df['Video'] = video_name
    all_summaries.append(df)

# STEP 4 — Combine and reshape
df_all = pd.concat(all_summaries, ignore_index=True)
df_long = df_all.melt(
    id_vars=["Metric", "Video"],
    value_vars=["FaceIt", "FaceMap"],
    var_name="Pipeline",
    value_name="Value"
)

# STEP 5 — Define metrics and color palette
metrics_to_plot = [
    "Mean Center Error",
    "RMSE (Area)",
    "Std Area Error",
    "Mean % Area Error",
    "Std Center Error"
]

palette_Center_Error = {"FaceIt": "#e67e22", "FaceMap": "#f4d03f" } #Orange, yellow
paletteRMSE = { "FaceIt": "#633974",  "FaceMap": "#c39bd3"} #violet
palette_Std_Area = { "FaceIt": "#1a5276",  "FaceMap": "#7fb3d5"} #blue
palette_Mean_perc_Area = { "FaceIt": "#117864",  "FaceMap": "#73c6b6"} #green
palette_ST_Center_Error = {"FaceIt": "#7b241c", "FaceMap": "#ec7063" } #brown

general_pallet = [palette_Center_Error, paletteRMSE, palette_Std_Area, palette_Mean_perc_Area, palette_ST_Center_Error]

# STEP 6 — Plot function
def plot_metric_boxplot(data, metric, palette, save_dir):
    subset = data[data["Metric"] == metric]
    faceit_vals = subset[subset["Pipeline"] == "FaceIt"]["Value"]
    facemap_vals = subset[subset["Pipeline"] == "FaceMap"]["Value"]

    # T-test
    t_stat, p_value = ttest_ind(faceit_vals, facemap_vals, equal_var=False)

    plt.figure(figsize=(7, 5))

    # Boxplot
    sns.boxplot(
        data=subset, x="Pipeline", y="Value",
        palette=palette, width=0.5,
        boxprops=dict(alpha=0.7), fliersize=0
    )

    # Colored Stripplot (matching but solid tone)
    for pipeline in subset["Pipeline"].unique():
        pipe_data = subset[subset["Pipeline"] == pipeline]
        sns.stripplot(
            data=pipe_data,
            x="Pipeline", y="Value",
            color=palette[pipeline],  # Same tone
            jitter=True, size=10, alpha=1.0, linewidth=0
        )

    # Annotate T-test result
    y_max = subset["Value"].max()
    y_min = subset["Value"].min()
    range_ = y_max - y_min
    height = y_max + range_ * 0.08
    line_y = y_max + range_ * 0.05

    # Significance stars
    if p_value < 0.0001:
        significance = "****"
    elif p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"  # not significant

    # Bracket with stars
    plt.plot([0, 0, 1, 1], [line_y, height, height, line_y], lw=1.5, c='black')
    plt.text(0.5, height + 0.01 * (y_max - y_min), significance, ha='center', fontsize=12)

    # Format
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Pipeline", fontsize=12)
    plt.title(f"{metric} (p = {p_value:.4f})", fontsize=13)

    plt.grid(False)
    plt.tight_layout()

    # Save
    filename = metric.replace(" ", "_").lower()
    png_path = os.path.join(save_dir, f"{filename}.png")
    svg_path = os.path.join(save_dir, f"{filename}.svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)
    plt.close()

    return {"Metric": metric, "T-statistic": t_stat, "P-value": p_value}

# STEP 7 — Plot each metric and collect t-test results
ttest_results = []
for metric in range(len(metrics_to_plot)):
    result = plot_metric_boxplot(df_long, metrics_to_plot[metric], general_pallet[metric], summary_dir)
    ttest_results.append(result)

# STEP 8 — Save stats and T-tests to Excel
df_filtered = df_long[df_long["Metric"] != "R2 (Area)"]
grouped = df_filtered.groupby(["Metric", "Pipeline"])["Value"].agg(["mean", "std", "min", "max"]).reset_index()
df_ttests = pd.DataFrame(ttest_results)

excel_path = os.path.join(summary_dir, "summary_across_trials.xlsx")
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    grouped.to_excel(writer, index=False, sheet_name="Grouped Stats")
    df_ttests.to_excel(writer, index=False, sheet_name="T-Test Results")

print(f"\n✅ Summary Excel saved to:\n{excel_path}")
print(f"✅ All plots saved to:\n{summary_dir}")
