from pathlib import Path

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import (
    kruskal,
    spearmanr,
    mannwhitneyu,
)

from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. 경로 설정
# ============================================================

ROOT = Path(
    r"C:\Users\park_younguk\Desktop\skin"
)

CSV_PATH = (
    ROOT
    / "lesion_background_color_texture_features.csv"
)

OUT_DIR = (
    ROOT
    / "analysis_results_color_texture"
)

OUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# 2. 결과 저장 폴더
# ============================================================

BOXPLOT_DIR = (
    OUT_DIR
    / "boxplots"
)

COLOR_PLOT_DIR = (
    BOXPLOT_DIR
    / "color"
)

TEXTURE_PLOT_DIR = (
    BOXPLOT_DIR
    / "texture"
)


COLOR_PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEXTURE_PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. 데이터 로드
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# 4. Severity 설정
# ============================================================

severity_order = [
    "mild",
    "moderate",
    "severe",
    "very severe",
]


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


df["severity_num"] = (
    df["severity"]
    .map(severity_map)
    .astype(float)
)


# ============================================================
# 데이터 기본 정보 출력
# ============================================================

print(
    "Total images:",
    len(df)
)

print()

print(
    df["severity"]
    .value_counts()
    .reindex(severity_order)
)


# ============================================================
# 5. Color Features
#
# lesion:
#   병변 자체 색상
#
# bg:
#   병변 제외 주변 피부 색상
#
# delta:
#   lesion - background
#
# abs_delta:
#   |lesion - background|
#
# wasserstein:
#   lesion과 background의
#   전체 pixel-value distribution 차이
# ============================================================

COLOR_FEATURES = [

    # --------------------------------------------------------
    # Lesion
    # --------------------------------------------------------

    "lesion_L_mean",
    "lesion_a_mean",
    "lesion_b_mean",
    "lesion_ITA_mean",


    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    "bg_L_mean",
    "bg_a_mean",
    "bg_b_mean",
    "bg_ITA_mean",


    # --------------------------------------------------------
    # Directional difference
    # lesion - background
    # --------------------------------------------------------

    "delta_L_mean",
    "delta_a_mean",
    "delta_b_mean",
    "delta_ITA_mean",


    # --------------------------------------------------------
    # Absolute difference
    # |lesion - background|
    # --------------------------------------------------------

    "abs_delta_L_mean",
    "abs_delta_a_mean",
    "abs_delta_b_mean",
    "abs_delta_ITA_mean",


    # --------------------------------------------------------
    # Wasserstein distance
    # --------------------------------------------------------

    "wasserstein_L",
    "wasserstein_a",
    "wasserstein_b",
    "wasserstein_ITA",
]


# ============================================================
# 6. Texture Features
#
# Texture:
#   RGB → Grayscale
#
# GLCM:
#   contrast
#   homogeneity
#   energy
#   correlation
#   entropy
#
# LBP:
#   entropy
#
# Texture에서는:
#   lesion
#   background
#   delta
#
# 만 사용한다.
#
# abs_delta는 사용하지 않는다.
# ============================================================

TEXTURE_FEATURES = [

    # --------------------------------------------------------
    # Lesion texture
    # --------------------------------------------------------

    "lesion_glcm_contrast",
    "lesion_glcm_homogeneity",
    "lesion_glcm_energy",
    "lesion_glcm_correlation",
    "lesion_glcm_entropy",
    "lesion_lbp_entropy",


    # --------------------------------------------------------
    # Background texture
    # --------------------------------------------------------

    "bg_glcm_contrast",
    "bg_glcm_homogeneity",
    "bg_glcm_energy",
    "bg_glcm_correlation",
    "bg_glcm_entropy",
    "bg_lbp_entropy",


    # --------------------------------------------------------
    # Texture difference
    # lesion - background
    # --------------------------------------------------------

    "delta_glcm_contrast",
    "delta_glcm_homogeneity",
    "delta_glcm_energy",
    "delta_glcm_correlation",
    "delta_glcm_entropy",
    "delta_lbp_entropy",
]


# ============================================================
# 7. 실제 CSV에 존재하는 Feature만 사용
# ============================================================

COLOR_FEATURES = [
    feature
    for feature in COLOR_FEATURES
    if feature in df.columns
]


TEXTURE_FEATURES = [
    feature
    for feature in TEXTURE_FEATURES
    if feature in df.columns
]


features = (
    COLOR_FEATURES
    + TEXTURE_FEATURES
)


# ============================================================
# Feature 개수 확인
# ============================================================

print()
print("=" * 60)

print(
    "Color features   :",
    len(COLOR_FEATURES)
)

print(
    "Texture features :",
    len(TEXTURE_FEATURES)
)

print(
    "Total features   :",
    len(features)
)

print("=" * 60)


# ============================================================
# 예상 Feature 개수 확인
#
# Color   = 20
# Texture = 18
# Total   = 38
# ============================================================

if len(COLOR_FEATURES) != 20:

    print(
        f"[WARNING] "
        f"Expected 20 color features, "
        f"found {len(COLOR_FEATURES)}"
    )


if len(TEXTURE_FEATURES) != 18:

    print(
        f"[WARNING] "
        f"Expected 18 texture features, "
        f"found {len(TEXTURE_FEATURES)}"
    )


if len(features) != 38:

    print(
        f"[WARNING] "
        f"Expected 38 total features, "
        f"found {len(features)}"
    )


# ============================================================
# 8. Feature type
#
# 결과 CSV에서
# Color / Texture 구분용
# ============================================================

feature_type_map = {

    **{
        feature: "color"
        for feature in COLOR_FEATURES
    },

    **{
        feature: "texture"
        for feature in TEXTURE_FEATURES
    },
}


# ============================================================
# 9. 중증도별 요약 통계
#
# count:
# 해당 feature 분석에 실제 사용 가능한 표본 수
#
# 예:
# lesion feature
# → mo_184 포함 가능
#
# background-dependent feature
# → mo_184는 NaN이므로 count에서 제외
# ============================================================

summary = (
    df
    .groupby(
        "severity",
        observed=False
    )[features]
    .agg([
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max",
    ])
)


summary_path = (
    OUT_DIR
    / "summary_by_severity.csv"
)


summary.to_csv(
    summary_path,
    encoding="utf-8-sig"
)


print(
    "Saved:",
    summary_path
)


# ============================================================
# 10. Kruskal-Wallis
#
# H0:
# 네 중증도 그룹의 분포가 동일하다.
#
# H1:
# 적어도 하나의 중증도 그룹이 다르다.
#
# NaN은 각 feature별로 제외한다.
# ============================================================

kruskal_rows = []


for feat in features:

    groups = [

        df.loc[
            df["severity"] == severity,
            feat
        ].dropna()

        for severity
        in severity_order
    ]


    # --------------------------------------------------------
    # 모든 severity group에 값이 존재하는 경우만 검정
    # --------------------------------------------------------

    if all(
        len(group) > 0
        for group in groups
    ):

        stat, p = kruskal(
            *groups
        )

    else:

        stat = np.nan
        p = np.nan


    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    kruskal_rows.append({

        "feature_type":
            feature_type_map[feat],

        "feature":
            feat,

        "n_mild":
            len(groups[0]),

        "n_moderate":
            len(groups[1]),

        "n_severe":
            len(groups[2]),

        "n_very_severe":
            len(groups[3]),

        "n_total":
            sum(
                len(group)
                for group in groups
            ),

        "kruskal_H":
            stat,

        "p_value":
            p,
    })


kruskal_df = pd.DataFrame(
    kruskal_rows
)


# ============================================================
# Kruskal FDR correction
# ============================================================

kruskal_df[
    "p_fdr"
] = np.nan


valid_p = (
    kruskal_df["p_value"]
    .notna()
)


if valid_p.any():

    kruskal_df.loc[
        valid_p,
        "p_fdr"
    ] = multipletests(

        kruskal_df.loc[
            valid_p,
            "p_value"
        ],

        method="fdr_bh"

    )[1]


kruskal_df[
    "significant_fdr"
] = (
    kruskal_df["p_fdr"] < 0.05
)


# FDR p-value가 작은 순으로 정렬
kruskal_df = (
    kruskal_df
    .sort_values(
        "p_fdr",
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


kruskal_path = (
    OUT_DIR
    / "kruskal_results.csv"
)


kruskal_df.to_csv(
    kruskal_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved:",
    kruskal_path
)


# ============================================================
# 11. Spearman correlation
#
# Severity:
# mild          = 0
# moderate      = 1
# severe        = 2
# very severe   = 3
#
# rho > 0:
# 중증도가 높아질수록 feature 증가
#
# rho < 0:
# 중증도가 높아질수록 feature 감소
#
# NaN은 해당 feature 분석에서만 제외
# ============================================================

spearman_rows = []


for feat in features:

    temp = (
        df[
            [
                "severity_num",
                feat,
            ]
        ]
        .dropna()
    )


    if len(temp) > 2:

        rho, p = spearmanr(
            temp["severity_num"],
            temp[feat]
        )

    else:

        rho = np.nan
        p = np.nan


    spearman_rows.append({

        "feature_type":
            feature_type_map[feat],

        "feature":
            feat,

        "n":
            len(temp),

        "spearman_rho":
            rho,

        "p_value":
            p,
    })


spearman_df = pd.DataFrame(
    spearman_rows
)


# ============================================================
# Spearman FDR correction
# ============================================================

spearman_df[
    "p_fdr"
] = np.nan


valid_p = (
    spearman_df["p_value"]
    .notna()
)


if valid_p.any():

    spearman_df.loc[
        valid_p,
        "p_fdr"
    ] = multipletests(

        spearman_df.loc[
            valid_p,
            "p_value"
        ],

        method="fdr_bh"

    )[1]


spearman_df[
    "significant_fdr"
] = (
    spearman_df["p_fdr"] < 0.05
)


# ------------------------------------------------------------
# Spearman rho 절댓값
#
# 방향과 관계없이
# severity와 강하게 관련된 feature 확인용
# ------------------------------------------------------------

spearman_df[
    "abs_spearman_rho"
] = (
    spearman_df[
        "spearman_rho"
    ].abs()
)


# ------------------------------------------------------------
# |rho|가 큰 순서로 정렬
# ------------------------------------------------------------

spearman_df = (
    spearman_df
    .sort_values(
        "abs_spearman_rho",
        ascending=False,
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


spearman_path = (
    OUT_DIR
    / "spearman_results.csv"
)


spearman_df.to_csv(
    spearman_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved:",
    spearman_path
)


# ============================================================
# 12. Pairwise Mann-Whitney U test
#
# 각 severity 쌍의 차이 확인
#
# mild vs moderate
# mild vs severe
# mild vs very severe
# moderate vs severe
# moderate vs very severe
# severe vs very severe
#
# NaN은 해당 feature 비교에서만 제외
# ============================================================

pairwise_rows = []


for feat in features:

    for i in range(
        len(severity_order)
    ):

        for j in range(
            i + 1,
            len(severity_order)
        ):

            g1_name = (
                severity_order[i]
            )

            g2_name = (
                severity_order[j]
            )


            # ------------------------------------------------
            # Group 1
            # ------------------------------------------------

            g1 = df.loc[
                df["severity"] == g1_name,
                feat
            ].dropna()


            # ------------------------------------------------
            # Group 2
            # ------------------------------------------------

            g2 = df.loc[
                df["severity"] == g2_name,
                feat
            ].dropna()


            # ------------------------------------------------
            # Mann-Whitney U
            # ------------------------------------------------

            if (
                len(g1) > 0
                and len(g2) > 0
            ):

                stat, p = mannwhitneyu(
                    g1,
                    g2,
                    alternative="two-sided"
                )

            else:

                stat = np.nan
                p = np.nan


            # ------------------------------------------------
            # 결과 저장
            # ------------------------------------------------

            pairwise_rows.append({

                "feature_type":
                    feature_type_map[feat],

                "feature":
                    feat,

                "group1":
                    g1_name,

                "group2":
                    g2_name,

                "n_group1":
                    len(g1),

                "n_group2":
                    len(g2),

                "U_stat":
                    stat,

                "p_value":
                    p,
            })


pairwise_df = pd.DataFrame(
    pairwise_rows
)


# ============================================================
# Pairwise FDR correction
#
# 현재는 모든 pairwise test를
# 하나의 multiple-testing family로 보고
# BH-FDR correction 적용
# ============================================================

pairwise_df[
    "p_fdr"
] = np.nan


valid_p = (
    pairwise_df["p_value"]
    .notna()
)


if valid_p.any():

    pairwise_df.loc[
        valid_p,
        "p_fdr"
    ] = multipletests(

        pairwise_df.loc[
            valid_p,
            "p_value"
        ],

        method="fdr_bh"

    )[1]


pairwise_df[
    "significant_fdr"
] = (
    pairwise_df["p_fdr"] < 0.05
)


pairwise_path = (
    OUT_DIR
    / "pairwise_mannwhitney_results.csv"
)


pairwise_df.to_csv(
    pairwise_path,
    index=False,
    encoding="utf-8-sig"
)


print(
    "Saved:",
    pairwise_path
)


# ============================================================
# 13. Boxplot
#
# Color / Texture 폴더를 분리해서 저장한다.
#
# seaborn은 해당 feature의 NaN을
# 자동으로 제외하고 그림을 생성한다.
# ============================================================

for feat in features:

    # --------------------------------------------------------
    # 저장 폴더 결정
    # --------------------------------------------------------

    if feat in COLOR_FEATURES:

        plot_dir = (
            COLOR_PLOT_DIR
        )

    else:

        plot_dir = (
            TEXTURE_PLOT_DIR
        )


    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    severity_palette = {
    "mild": "#D9EAF7",
    "moderate": "#9ECAE1",
    "severe": "#4292C6",
    "very severe": "#084594",
}

for feat in features:

    if feat in COLOR_FEATURES:
        plot_dir = COLOR_PLOT_DIR
    else:
        plot_dir = TEXTURE_PLOT_DIR

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="severity",
        y=feat,
        order=severity_order,
        palette=severity_palette,
        showfliers=True,   # 이상치만 점으로 표시
        width=0.6,
        linewidth=1.5
    )

    #plt.title(feat)
    plt.xlabel("Severity")
    plt.ylabel(feat)
    plt.tight_layout()

    out_path = plot_dir / f"boxplot_{feat}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


print(
    "Boxplots saved."
)


# ============================================================
# 14. 분석 종료
# ============================================================

print()
print("=" * 60)
print("Statistical analysis completed")
print("=" * 60)


print(
    f"Total images              : "
    f"{len(df)}"
)


print(
    f"Color features analyzed   : "
    f"{len(COLOR_FEATURES)}"
)


print(
    f"Texture features analyzed : "
    f"{len(TEXTURE_FEATURES)}"
)


print(
    f"Total features analyzed   : "
    f"{len(features)}"
)


print(
    f"Output directory          : "
    f"{OUT_DIR}"
)