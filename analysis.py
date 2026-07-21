from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal, spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(r"C:\Users\park_younguk\Desktop\skin")
CSV_PATH = ROOT / "lesion_background_ita_lab_features.csv"
OUT_DIR = ROOT / "analysis_results"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

severity_order = ["mild", "moderate", "severe", "very severe"]
severity_map = {
    "mild": 0,
    "moderate": 1,
    "severe": 2,
    "very severe": 3,
}

df["severity"] = pd.Categorical(
    df["severity"],
    categories=severity_order,
    ordered=True
)
df["severity_num"] = df["severity"].map(severity_map)

print("Total images:", len(df))
print(df["severity"].value_counts().reindex(severity_order))

'''
lesion: 병변 자체의 색상
bg: 병변 제외 배경의 색상
delta: 병변과 배경의 단순 차이
abs: 병변과 배경의 절대 차이
wass: 병변가 배경이 색상 히스토그램 분포 거리
'''
#  feature
features = [
    
    "lesion_L_mean",
    "lesion_a_mean",
    "lesion_b_mean",
    "lesion_ITA_mean",
    "bg_L_mean",
    "bg_a_mean",
    "bg_b_mean",
    "bg_ITA_mean",
    "delta_L_mean",
    "delta_a_mean",
    "delta_b_mean",
    "delta_ITA_mean",
    "abs_delta_L_mean",
    "abs_delta_a_mean",
    "abs_delta_b_mean",
    "abs_delta_ITA_mean",
    "wasserstein_L",
    "wasserstein_a",
    "wasserstein_b",
    "wasserstein_ITA"
]

features = [f for f in features if f in df.columns]


# 중증도별 요약 통계 

summary = df.groupby("severity", observed=False)[features].agg(
    ["mean", "std", "median", "min", "max"]
)

summary_path = OUT_DIR / "summary_by_severity.csv"
summary.to_csv(summary_path, encoding="utf-8-sig")
print("Saved:", summary_path)


# Kruskal-Wallis 검정
kruskal_rows = []

for feat in features:
    groups = [
        df.loc[df["severity"] == sev, feat].dropna()
        for sev in severity_order
    ]

    if all(len(g) > 0 for g in groups):
        stat, p = kruskal(*groups)
    else:
        stat, p = np.nan, np.nan

    kruskal_rows.append({
        "feature": feat,
        "kruskal_H": stat,
        "p_value": p,
    })

kruskal_df = pd.DataFrame(kruskal_rows)

# 다중검정 보정 FDR
valid_p = kruskal_df["p_value"].notna()
kruskal_df.loc[valid_p, "p_fdr"] = multipletests(
    kruskal_df.loc[valid_p, "p_value"],
    method="fdr_bh"
)[1]

kruskal_path = OUT_DIR / "kruskal_results.csv"
kruskal_df.to_csv(kruskal_path, index=False, encoding="utf-8-sig")
print("Saved:", kruskal_path)


# Spearman 상관분석 -> 중증도가 심해질수록 이 색상 수치도 같이 커지거나 작아지는 지 확인.
spearman_rows = []

for feat in features:
    temp = df[["severity_num", feat]].dropna()

    if len(temp) > 2:
        rho, p = spearmanr(temp["severity_num"], temp[feat])
    else:
        rho, p = np.nan, np.nan

    spearman_rows.append({
        "feature": feat,
        "spearman_rho": rho,
        "p_value": p,
    })

spearman_df = pd.DataFrame(spearman_rows)

valid_p = spearman_df["p_value"].notna()
spearman_df.loc[valid_p, "p_fdr"] = multipletests(
    spearman_df.loc[valid_p, "p_value"],
    method="fdr_bh"
)[1]

spearman_path = OUT_DIR / "spearman_results.csv"
spearman_df.to_csv(spearman_path, index=False, encoding="utf-8-sig")
print("Saved:", spearman_path)


# Pairwise Mann-Whitney U test  -> 1:1로 비교했을 때도 차이가 나는지 확인
pairwise_rows = []

for feat in features:
    for i in range(len(severity_order)):
        for j in range(i + 1, len(severity_order)):
            g1_name = severity_order[i]
            g2_name = severity_order[j]

            g1 = df.loc[df["severity"] == g1_name, feat].dropna()
            g2 = df.loc[df["severity"] == g2_name, feat].dropna()

            if len(g1) > 0 and len(g2) > 0:
                stat, p = mannwhitneyu(g1, g2, alternative="two-sided")
            else:
                stat, p = np.nan, np.nan

            pairwise_rows.append({
                "feature": feat,
                "group1": g1_name,
                "group2": g2_name,
                "U_stat": stat,
                "p_value": p,
            })

pairwise_df = pd.DataFrame(pairwise_rows)

valid_p = pairwise_df["p_value"].notna()
pairwise_df.loc[valid_p, "p_fdr"] = multipletests(
    pairwise_df.loc[valid_p, "p_value"],
    method="fdr_bh"
)[1]

pairwise_path = OUT_DIR / "pairwise_mannwhitney_results.csv"
pairwise_df.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
print("Saved:", pairwise_path)

# Boxplot 저장
plot_features = features
    
plot_features = [f for f in plot_features if f in df.columns]

for feat in plot_features:
    plt.figure(figsize=(7, 5))
    sns.boxplot(
        data=df,
        x="severity",
        y=feat,
        order=severity_order,
        showfliers=False
    )
    sns.stripplot(
        data=df,
        x="severity",
        y=feat,
        order=severity_order,
        color="black",
        alpha=0.35,
        size=2
    )

    plt.title(feat)
    plt.xlabel("Severity")
    plt.ylabel(feat)
    plt.tight_layout()

    out_path = OUT_DIR / f"boxplot_{feat}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)

print("Analysis done.")