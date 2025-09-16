#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import ttest_ind
# def Summery_compile_per_condition():
#     excel_paths = [
#         r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\series_1\result2\pupil_detection_results.xlsx",
#         r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\series_2\result2\pupil_detection_results.xlsx",
#         r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\series_3\result2\pupil_detection_results.xlsx",
#         r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\series_4\result2\pupil_detection_results.xlsx",
#         r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\series_5\result2\pupil_detection_results.xlsx",
#     ]
#
#     all_data = []
#     for path in excel_paths:
#         session_name = os.path.basename(os.path.dirname(os.path.dirname(path)))  # e.g., series_1
#         df = pd.read_excel(path, sheet_name="Summary")
#         df["Session"] = session_name
#         all_data.append(df)
#
#     # Combine into one DataFrame
#     df_all = pd.concat(all_data, ignore_index=True)
#
#     # Reshape into long format
#     df_long = df_all.melt(
#         id_vars=["Metric", "Session"],
#         value_vars=["FaceIt", "FaceMap"],
#         var_name="Pipeline",
#         value_name="Value"
#     )
#
#     # Add Condition info (optional, useful for combining across conditions later)
#     df_long["Condition"] = "Condition7"
#
#     # Save compiled version (optional)
#     output_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\7_contrast_change\compiled_summary_condition7.xlsx"
#     df_long.to_excel(output_path, index=False)
#
#     print(f"✅ Compiled data saved to:\n{output_path}")
# Summery_compile_per_condition()
#
#
# def pval_to_star(p):
#     if p < 0.0001: return "****"
#     elif p < 0.001: return "***"
#     elif p < 0.01: return "**"
#     elif p < 0.05: return "*"
#     else: return "ns"
#
# def plot_metric_across_conditions(
#     compiled_paths: dict,     # {Condition: path}
#     metric: str,              # e.g., "RMSE (Area)"
#     save_dir: str             # where to save the plots
# ):
#     os.makedirs(save_dir, exist_ok=True)
#     all_data = []
#
#     for condition, path in compiled_paths.items():
#         df = pd.read_excel(path)
#         all_data.append(df[df["Metric"] == metric])
#
#     df_combined = pd.concat(all_data, ignore_index=True)
#
#     # Plot
#     plt.figure(figsize=(10, 6))
#     palette = {"FaceIt": "#4C72B0", "FaceMap": "#DD8452"}
#
#     sns.boxplot(data=df_combined, x="Condition", y="Value", hue="Pipeline",
#                 palette=palette, width=0.6, fliersize=0)
#
#     sns.stripplot(data=df_combined, x="Condition", y="Value", hue="Pipeline",
#                   dodge=True, palette=palette, alpha=0.7, size=6, linewidth=0)
#
#     # Remove duplicate legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     plt.legend(handles[:2], labels[:2], title="Pipeline")
#
#     # Significance annotations
#     for i, condition in enumerate(df_combined["Condition"].unique()):
#         subset = df_combined[df_combined["Condition"] == condition]
#         vals_faceit = subset[subset["Pipeline"] == "FaceIt"]["Value"]
#         vals_facemap = subset[subset["Pipeline"] == "FaceMap"]["Value"]
#         if len(vals_faceit) > 0 and len(vals_facemap) > 0:
#             t_stat, p_val = ttest_ind(vals_faceit, vals_facemap, equal_var=False)
#             star = pval_to_star(p_val)
#
#             y_max = subset["Value"].max()
#             y_min = subset["Value"].min()
#             height = y_max + (y_max - y_min) * 0.25
#             line_y = height - (y_max - y_min) * 0.05
#
#             plt.plot([i - 0.2, i - 0.2, i + 0.2, i + 0.2],
#                      [line_y, height, height, line_y],
#                      c="black", lw=1.3)
#             plt.text(i, height + 0.01 * (y_max - y_min), star,
#                      ha='center', fontsize=12)
#
#     plt.title(f"{metric} Across Conditions", fontsize=14)
#     plt.ylabel(metric, fontsize=12)
#     plt.xlabel("Condition", fontsize=12)
#     plt.tight_layout()
#
#     safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "").lower()
#     plt.savefig(os.path.join(save_dir, f"{safe_metric}.png"), dpi=300)
#     plt.savefig(os.path.join(save_dir, f"{safe_metric}.svg"))
#     plt.close()
#
#     print(f"✅ Saved plot for {metric} to: {save_dir}")
# # One-time setup
# base_dir = r"C:\Users\faezeh.rabbani\Desktop\FACEIT"
# compiled_paths = {
#     "Condition6": os.path.join(base_dir, "6_random_trajectory_dilated_reflection_shadow", "compiled_summary_condition6.xlsx"),
#     "Condition7": os.path.join(base_dir, "7_contrast_change", "compiled_summary_condition7.xlsx"),
#     "Condition8": os.path.join(base_dir, "8_contrast_change_shadow", "compiled_summary_condition8.xlsx"),
# }
#
# # Call for one or more metrics
# output_dir = os.path.join(base_dir, "final_metric_plots")
# plot_metric_across_conditions(compiled_paths, "RMSE (Area)", output_dir)
# plot_metric_across_conditions(compiled_paths, "Mean Center Error", output_dir)
# plot_metric_across_conditions(compiled_paths, "Mean % Area Error", output_dir)
# plot_metric_across_conditions(compiled_paths, "Std Center Error", output_dir)
# plot_metric_across_conditions(compiled_paths, "Std Area Error", output_dir)

import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ---------- Compile per condition (now includes DLC) ----------
def Summery_compile_per_condition():
    excel_paths = [
        r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\series_1\result2\pupil_detection_results.xlsx",
        r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\series_2\result2\pupil_detection_results.xlsx",
        r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\series_3\result2\pupil_detection_results.xlsx",
        r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\series_4\result2\pupil_detection_results.xlsx",
        r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\series_5\result2\pupil_detection_results.xlsx",
    ]

    all_data = []
    for path in excel_paths:
        session_name = os.path.basename(os.path.dirname(os.path.dirname(path)))  # e.g., series_1
        df = pd.read_excel(path, sheet_name="Summary")

        # keep only known metric columns that exist in this file
        available_pipelines = [c for c in ["FaceIt", "FaceMap", "DLC"] if c in df.columns]
        if not available_pipelines:
            continue

        df["Session"] = session_name

        # reshape to long
        df_long = df.melt(
            id_vars=["Metric", "Session"],
            value_vars=available_pipelines,
            var_name="Pipeline",
            value_name="Value"
        )
        df_long["Condition"] = "Condition7"
        all_data.append(df_long)

    if not all_data:
        raise RuntimeError("No data found to compile.")

    df_all = pd.concat(all_data, ignore_index=True)

    # Save compiled version
    output_path = r"C:\Users\faezeh.rabbani\Desktop\FACEIT\Condition_2.fix_calibration_dilation\compiled_summary_condition2.xlsx"
    df_all.to_excel(output_path, index=False)
    print(f"✅ Compiled data saved to:\n{output_path}")

Summery_compile_per_condition()


# ---------- Plotting helpers ----------
def pval_to_star(p):
    if p < 1e-4: return "****"
    elif p < 1e-3: return "***"
    elif p < 1e-2: return "**"
    elif p < 5e-2: return "*"
    else: return "ns"


def plot_metric_across_conditions(
    compiled_paths: dict,     # {Condition: path}
    metric: str,              # e.g., "RMSE (Area)"
    save_dir: str
):
    os.makedirs(save_dir, exist_ok=True)
    all_data = []

    # Load & filter metric
    for condition, path in compiled_paths.items():
        df = pd.read_excel(path)
        df = df[df["Metric"] == metric].copy()
        # ensure condition label matches the dict key (in case file includes a different label)
        df["Condition"] = condition
        all_data.append(df)

    df_combined = pd.concat(all_data, ignore_index=True)

    # Color palette (only for pipelines that are present)
    pipelines_present = sorted(df_combined["Pipeline"].unique())
    default_palette = {
        "FaceIt":  "#4C72B0",
        "FaceMap": "#DD8452",
        "DLC":     "#55A868",
    }
    palette = {k: default_palette.get(k, None) for k in pipelines_present}

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_combined, x="Condition", y="Value", hue="Pipeline",
        palette=palette, width=0.6, fliersize=0
    )
    sns.stripplot(
        data=df_combined, x="Condition", y="Value", hue="Pipeline",
        dodge=True, palette=palette, alpha=0.7, size=6, linewidth=0
    )

    # Remove duplicated legend (from box + strip)
    handles, labels = ax.get_legend_handles_labels()
    # keep only first set of unique labels in original order
    seen, uniq = set(), []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq.append((h, l))
    ax.legend([h for h, _ in uniq], [l for _, l in uniq], title="Pipeline")

    # Pairwise significance per condition for all pipeline pairs present
    pairs = list(itertools.combinations(pipelines_present, 2))
    for i, condition in enumerate(df_combined["Condition"].unique()):
        subset = df_combined[df_combined["Condition"] == condition]
        y_min, y_max = subset["Value"].min(), subset["Value"].max()
        span = max(1e-9, (y_max - y_min))

        # vertical stacking for multiple pairs
        base = y_max + 0.15 * span
        step = 0.12 * span

        for j, (p1, p2) in enumerate(pairs):
            v1 = subset[subset["Pipeline"] == p1]["Value"].dropna()
            v2 = subset[subset["Pipeline"] == p2]["Value"].dropna()
            if len(v1) > 0 and len(v2) > 0:
                _, p_val = ttest_ind(v1, v2, equal_var=False)
                star = pval_to_star(p_val)

                height = base + j * step
                line_y = height - 0.03 * span
                # small horizontal jitter to draw multiple brackets clearly
                offset = (-0.22 + j*0.22) if len(pairs) > 1 else 0.0
                x_left, x_right = i - 0.25 + offset, i + 0.25 + offset

                plt.plot([x_left, x_left, x_right, x_right],
                         [line_y, height, height, line_y],
                         c="black", lw=1.2)
                plt.text((x_left + x_right) / 2, height + 0.01 * span,
                         f"{p1} vs {p2}: {star}", ha='center', fontsize=10)

    plt.title(f"{metric} Across Conditions", fontsize=14)
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Condition", fontsize=12)
    plt.tight_layout()

    safe_metric = metric.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plt.savefig(os.path.join(save_dir, f"{safe_metric}.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, f"{safe_metric}.svg"))
    plt.close()

    print(f"✅ Saved plot for {metric} to: {save_dir}")


# ---------- One-time setup ----------
base_dir = r"C:\Users\faezeh.rabbani\Desktop\FACEIT"
compiled_paths = {
    # "Condition6": os.path.join(base_dir, "6_random_trajectory_dilated_reflection_shadow", "compiled_summary_condition6.xlsx"),
    "Condition1": os.path.join(base_dir, "Condition_2.fix_calibration_dilation", "compiled_summary_condition2.xlsx"),
    # "Condition8": os.path.join(base_dir, "8_contrast_change_shadow", "compiled_summary_condition8.xlsx"),
}

# ---------- Generate figures (now compares FaceIt, FaceMap, DLC) ----------
output_dir = os.path.join(base_dir, "final_metric_plots")
# plot_metric_across_conditions(compiled_paths, "RMSE (Area)", output_dir)
plot_metric_across_conditions(compiled_paths, "Mean Center Error", output_dir)
plot_metric_across_conditions(compiled_paths, "Mean % Area Error", output_dir)
plot_metric_across_conditions(compiled_paths, "Std Center Error", output_dir)
plot_metric_across_conditions(compiled_paths, "Std Area Error", output_dir)
