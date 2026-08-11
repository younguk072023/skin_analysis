from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.base import clone

from models.model_zoo import get_models

from features.feature_color import feature_color
from features.feature_texture import feature_texture
from features.feature_combined import feature_combined


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
    / "ml_results_color_texture"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 데이터 로드
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# 3. Severity 설정
# ============================================================

severity_order = [
    "mild",
    "moderate",
    "severe",
    "very severe",
]


df = df[
    df["severity"].isin(
        severity_order
    )
].copy()


# ============================================================
# 4. Label Encoding
# ============================================================

label_encoder = LabelEncoder()

label_encoder.fit(
    severity_order
)


print("=" * 60)
print("Dataset information")
print("=" * 60)

print(
    "Total images:",
    len(df)
)


print(
    "\nClass mapping:"
)


for i, cls in enumerate(
    label_encoder.classes_
):

    print(
        i,
        "->",
        cls
    )


print(
    "\nClass counts:"
)


print(
    df["severity"]
    .value_counts()
    .reindex(
        severity_order
    )
)


# ============================================================
# 5. Feature Group 통합
#
# Color
# Texture
# Combined
# ============================================================

# ------------------------------------------------------------
# 서로 같은 group 이름이 존재하는지 검사
# ------------------------------------------------------------

group_name_sets = [
    set(
        feature_color.keys()
    ),
    set(
        feature_texture.keys()
    ),
    set(
        feature_combined.keys()
    ),
]


duplicate_group_names = (

    (
        group_name_sets[0]
        & group_name_sets[1]
    )

    |

    (
        group_name_sets[0]
        & group_name_sets[2]
    )

    |

    (
        group_name_sets[1]
        & group_name_sets[2]
    )
)


if duplicate_group_names:

    raise ValueError(
        "중복된 Feature Group 이름이 존재합니다:\n"
        + "\n".join(
            sorted(
                duplicate_group_names
            )
        )
    )


# ------------------------------------------------------------
# 모든 Feature Group 통합
# ------------------------------------------------------------

feature_groups = {
    **feature_color,
    **feature_texture,
    **feature_combined,
}


# ============================================================
# 6. Feature Family 정의
#
# 결과에서
# color / texture / combined
# 구분하기 위함
# ============================================================

feature_family_map = {}


for group_name in feature_color.keys():

    feature_family_map[
        group_name
    ] = "color"


for group_name in feature_texture.keys():

    feature_family_map[
        group_name
    ] = "texture"


for group_name in feature_combined.keys():

    feature_family_map[
        group_name
    ] = "combined"


# ============================================================
# 7. CSV Feature 존재 여부 확인
# ============================================================

required_features = sorted({

    feature

    for feature_cols
    in feature_groups.values()

    for feature
    in feature_cols
})


missing_features = [

    feature

    for feature
    in required_features

    if feature
    not in df.columns
]


if missing_features:

    raise ValueError(
        "CSV에 다음 feature가 없습니다:\n"
        + "\n".join(
            missing_features
        )
    )


# ============================================================
# Feature Group 정보 출력
# ============================================================

print()
print("=" * 60)
print("Feature Group information")
print("=" * 60)


print(
    "Color groups    :",
    len(
        feature_color
    )
)


print(
    "Texture groups  :",
    len(
        feature_texture
    )
)


print(
    "Combined groups :",
    len(
        feature_combined
    )
)


print(
    "Total groups    :",
    len(
        feature_groups
    )
)


print(
    "Unique features :",
    len(
        required_features
    )
)


# ============================================================
# 8. 모델 불러오기
# ============================================================

models = get_models(
    random_state=42
)


print()
print(
    "Models:"
)


for model_name in models.keys():

    print(
        "-",
        model_name
    )


# ============================================================
# 9. Pipeline 내부의 최종 estimator 가져오기
#
# model_zoo에서 Pipeline을 사용하는 모델도
# feature_importances_를 읽을 수 있도록 처리
# ============================================================

def get_final_estimator(model):

    if hasattr(
        model,
        "steps"
    ):

        return model.steps[-1][1]

    return model


# ============================================================
# 10. Feature Group 평가 함수
# ============================================================

def evaluate_group(
    model_name,
    model,
    group_name,
    feature_cols
):

    print()
    print("=" * 70)

    print(
        "Model:",
        model_name
    )

    print(
        "Feature family:",
        feature_family_map[
            group_name
        ]
    )

    print(
        "Feature group:",
        group_name
    )

    print(
        "Number of features:",
        len(
            feature_cols
        )
    )

    print(
        "Features:"
    )

    for feature in feature_cols:

        print(
            "  -",
            feature
        )


    # ========================================================
    # 모델별 출력 폴더
    # ========================================================

    model_out_dir = (
        OUT_DIR
        / model_name
    )

    model_out_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 해당 Feature Group에서 필요한 열만 가져오기
    #
    # 중요:
    #
    # lesion-only group
    # → mo_184 사용 가능
    #
    # background / delta 포함 group
    # → mo_184의 NaN 때문에
    #   해당 group에서만 자동 제외
    # ========================================================

    data = (
        df[
            feature_cols
            + ["severity"]
        ]
        .dropna()
        .copy()
    )


    # ========================================================
    # 실제 분석에 사용되는 클래스별 N
    # ========================================================

    class_counts = (
        data["severity"]
        .value_counts()
        .reindex(
            severity_order,
            fill_value=0
        )
    )


    print()
    print(
        "Samples used:",
        len(
            data
        )
    )


    print(
        "Class counts:"
    )


    print(
        class_counts
    )


    # ========================================================
    # 5-fold CV 가능 여부 확인
    # ========================================================

    if (
        class_counts < 5
    ).any():

        raise ValueError(
            f"{model_name} / "
            f"{group_name}: "
            f"5-fold CV를 수행하기에는 "
            f"일부 클래스의 표본 수가 부족합니다.\n"
            f"{class_counts}"
        )


    # ========================================================
    # X / y
    # ========================================================

    X = (
        data[
            feature_cols
        ]
        .copy()
    )


    y_group = (
        label_encoder
        .transform(
            data[
                "severity"
            ]
        )
    )


    # ========================================================
    # 5-Fold Stratified CV
    # ========================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )


    # ========================================================
    # Fold별 평가 결과
    # ========================================================

    accs = []

    macro_f1s = []

    bal_accs = []


    # ========================================================
    # Out-of-fold prediction
    #
    # 전체 fold 예측을 합쳐서
    # confusion matrix / classification report 계산
    # ========================================================

    y_true_all = []

    y_pred_all = []


    # ========================================================
    # Feature Importance
    # ========================================================

    importance_list = []


    # ========================================================
    # 11. Fold 반복
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(
            X,
            y_group
        ),
        start=1
    ):


        # ----------------------------------------------------
        # Train / Test 분리
        # ----------------------------------------------------

        X_train = X.iloc[
            train_idx
        ]


        X_test = X.iloc[
            test_idx
        ]


        y_train = y_group[
            train_idx
        ]


        y_test = y_group[
            test_idx
        ]


        # ----------------------------------------------------
        # Fold마다 새 모델 생성
        # ----------------------------------------------------

        clf = clone(
            model
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        clf.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = clf.predict(
            X_test
        )


        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        acc = accuracy_score(
            y_test,
            y_pred
        )


        # ----------------------------------------------------
        # Macro F1
        #
        # 각 severity class를 동일 비중으로 평가
        # ----------------------------------------------------

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )


        # ----------------------------------------------------
        # Balanced Accuracy
        #
        # 각 class recall 평균
        # ----------------------------------------------------

        bal_acc = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )


        accs.append(
            acc
        )


        macro_f1s.append(
            macro_f1
        )


        bal_accs.append(
            bal_acc
        )


        # ----------------------------------------------------
        # Out-of-fold prediction
        # ----------------------------------------------------

        y_true_all.extend(
            y_test
        )


        y_pred_all.extend(
            y_pred
        )


        # ----------------------------------------------------
        # Feature importance
        # ----------------------------------------------------

        estimator = (
            get_final_estimator(
                clf
            )
        )


        if hasattr(
            estimator,
            "feature_importances_"
        ):

            importance_list.append(
                estimator.feature_importances_
            )


        # ----------------------------------------------------
        # Fold 결과 출력
        # ----------------------------------------------------

        print(
            f"Fold {fold}: "
            f"Acc={acc:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Acc={bal_acc:.4f}"
        )


    # ========================================================
    # 12. CV 결과 평균
    # ========================================================

    accuracy_mean = (
        np.mean(
            accs
        )
    )


    accuracy_std = (
        np.std(
            accs
        )
    )


    macro_f1_mean = (
        np.mean(
            macro_f1s
        )
    )


    macro_f1_std = (
        np.std(
            macro_f1s
        )
    )


    balanced_accuracy_mean = (
        np.mean(
            bal_accs
        )
    )


    balanced_accuracy_std = (
        np.std(
            bal_accs
        )
    )


    print()
    print(
        "5-fold CV results:"
    )


    print(
        f"Accuracy: "
        f"{accuracy_mean:.4f} "
        f"± "
        f"{accuracy_std:.4f}"
    )


    print(
        f"Macro F1: "
        f"{macro_f1_mean:.4f} "
        f"± "
        f"{macro_f1_std:.4f}"
    )


    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_mean:.4f} "
        f"± "
        f"{balanced_accuracy_std:.4f}"
    )


    # ========================================================
    # 13. Classification Report
    # ========================================================

    report_text = (
        classification_report(
            y_true_all,
            y_pred_all,
            labels=np.arange(
                len(
                    label_encoder.classes_
                )
            ),
            target_names=
                label_encoder.classes_,
            zero_division=0
        )
    )


    print()
    print(
        "Classification report "
        "from all folds:"
    )


    print(
        report_text
    )


    # ========================================================
    # Classification Report 저장
    # ========================================================

    report_path = (
        model_out_dir
        / f"classification_report_{group_name}.txt"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report_text
        )


    # ========================================================
    # 14. Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=np.arange(
            len(
                label_encoder.classes_
            )
        )
    )


    plt.figure(
        figsize=(6, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=
            label_encoder.classes_,
        yticklabels=
            label_encoder.classes_
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "True"
    )


    plt.title(
        f"{model_name} - "
        f"{group_name}"
    )


    plt.tight_layout()


    cm_path = (
        model_out_dir
        / f"confusion_matrix_{group_name}.png"
    )


    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # 15. Feature Importance
    #
    # Random Forest / ExtraTrees / XGBoost 등
    # feature_importances_가 존재할 때만 저장
    # ========================================================

    if (
        len(
            importance_list
        )
        > 0
    ):

        importance_df = (
            pd.DataFrame({

                "feature":
                    feature_cols,

                "importance_mean":
                    np.mean(
                        importance_list,
                        axis=0
                    ),

                "importance_std":
                    np.std(
                        importance_list,
                        axis=0
                    ),
            })
            .sort_values(
                "importance_mean",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        importance_path = (
            model_out_dir
            / f"feature_importance_{group_name}.csv"
        )


        importance_df.to_csv(
            importance_path,
            index=False,
            encoding="utf-8-sig"
        )


    # ========================================================
    # 16. 결과 반환
    # ========================================================

    return {

        "model":
            model_name,

        "feature_family":
            feature_family_map[
                group_name
            ],

        "feature_group":
            group_name,

        "n_samples":
            len(
                data
            ),

        "n_features":
            len(
                feature_cols
            ),

        "n_mild":
            int(
                class_counts[
                    "mild"
                ]
            ),

        "n_moderate":
            int(
                class_counts[
                    "moderate"
                ]
            ),

        "n_severe":
            int(
                class_counts[
                    "severe"
                ]
            ),

        "n_very_severe":
            int(
                class_counts[
                    "very severe"
                ]
            ),

        "accuracy_mean":
            accuracy_mean,

        "accuracy_std":
            accuracy_std,

        "macro_f1_mean":
            macro_f1_mean,

        "macro_f1_std":
            macro_f1_std,

        "balanced_accuracy_mean":
            balanced_accuracy_mean,

        "balanced_accuracy_std":
            balanced_accuracy_std,
    }


# ============================================================
# 17. 모든 Model × Feature Group 평가
# ============================================================

results = []


for model_name, model in models.items():

    for (
        group_name,
        feature_cols
    ) in feature_groups.items():


        # ----------------------------------------------------
        # 빈 그룹은 건너뜀
        # ----------------------------------------------------

        if len(
            feature_cols
        ) == 0:

            continue


        result = evaluate_group(
            model_name=
                model_name,

            model=
                model,

            group_name=
                group_name,

            feature_cols=
                feature_cols
        )


        results.append(
            result
        )


# ============================================================
# 18. 전체 결과 DataFrame
# ============================================================

results_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# Macro F1 기준으로 정렬
# ------------------------------------------------------------

results_df = (
    results_df
    .sort_values(
        "macro_f1_mean",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 19. 전체 결과 저장
# ============================================================

results_path = (
    OUT_DIR
    / "ml_group_comparison_results.csv"
)


results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)

print(
    "Saved group comparison:",
    results_path
)


print()
print(
    results_df
)


# ============================================================
# 20. 모델별 결과 저장
# ============================================================

for model_name in (
    results_df[
        "model"
    ].unique()
):


    model_df = (
        results_df[
            results_df[
                "model"
            ]
            == model_name
        ]
        .copy()
    )


    model_path = (
        OUT_DIR
        / model_name
        / "group_comparison_results.csv"
    )


    model_df.to_csv(
        model_path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 21. Feature Group 중심 Pivot Table
# ============================================================

print()
print("=" * 70)

print(
    "Feature Group 중심 결과 재정리 중..."
)


# ============================================================
# Macro F1 Mean
# ============================================================

pivot_f1_mean = (
    results_df
    .pivot(
        index="feature_group",
        columns="model",
        values="macro_f1_mean"
    )
)


# ============================================================
# Macro F1 Std
# ============================================================

pivot_f1_std = (
    results_df
    .pivot(
        index="feature_group",
        columns="model",
        values="macro_f1_std"
    )
)


# ============================================================
# Accuracy
# ============================================================

pivot_acc_mean = (
    results_df
    .pivot(
        index="feature_group",
        columns="model",
        values="accuracy_mean"
    )
)


# ============================================================
# Balanced Accuracy
# ============================================================

pivot_bal_acc_mean = (
    results_df
    .pivot(
        index="feature_group",
        columns="model",
        values="balanced_accuracy_mean"
    )
)


# ============================================================
# 22. Pivot 결과 저장
# ============================================================

pivot_f1_mean.to_csv(
    OUT_DIR
    / "pivot_macro_f1_mean.csv",
    encoding="utf-8-sig"
)


pivot_f1_std.to_csv(
    OUT_DIR
    / "pivot_macro_f1_std.csv",
    encoding="utf-8-sig"
)


pivot_acc_mean.to_csv(
    OUT_DIR
    / "pivot_accuracy_mean.csv",
    encoding="utf-8-sig"
)


pivot_bal_acc_mean.to_csv(
    OUT_DIR
    / "pivot_balanced_accuracy_mean.csv",
    encoding="utf-8-sig"
)


# ============================================================
# 23. 평균 ± 표준편차 종합표
# ============================================================

formatted_summary = pd.DataFrame(
    index=
        pivot_f1_mean.index
)


for model_name in (
    pivot_f1_mean.columns
):

    formatted_summary[
        f"{model_name} (Macro F1)"
    ] = (

        pivot_f1_mean[
            model_name
        ]
        .map(
            "{:.4f}".format
        )

        + " ± "

        + pivot_f1_std[
            model_name
        ]
        .map(
            "{:.4f}".format
        )
    )


# ============================================================
# Feature Family
# ============================================================

formatted_summary[
    "Feature_Family"
] = [

    feature_family_map[
        group_name
    ]

    for group_name
    in formatted_summary.index
]


# ============================================================
# 최고 성능 모델
# ============================================================

formatted_summary[
    "Best_Model"
] = (
    pivot_f1_mean
    .idxmax(
        axis=1
    )
)


formatted_summary[
    "Best_Macro_F1"
] = (
    pivot_f1_mean
    .max(
        axis=1
    )
)


# ============================================================
# 저장
# ============================================================

feature_centric_path = (
    OUT_DIR
    / "feature_centric_model_comparison.csv"
)


formatted_summary.to_csv(
    feature_centric_path,
    encoding="utf-8-sig"
)


print(
    "Saved feature-centric comparison:",
    feature_centric_path
)


# ============================================================
# 24. Feature Group × Model
# Macro F1 Heatmap
# ============================================================

heatmap_height = max(
    7,
    len(
        pivot_f1_mean
    )
    * 0.35
)


plt.figure(
    figsize=(
        11,
        heatmap_height
    )
)


sns.heatmap(
    pivot_f1_mean,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    cbar_kws={
        "label":
            "Mean Macro F1-score"
    }
)


plt.title(
    "Macro F1-score by Feature Group & Model",
    fontsize=14,
    pad=15
)


plt.xlabel(
    "Model",
    fontsize=12
)


plt.ylabel(
    "Feature Group",
    fontsize=12
)


plt.tight_layout()


heatmap_path = (
    OUT_DIR
    / "feature_centric_f1_heatmap.png"
)


plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved Feature Group heatmap:",
    heatmap_path
)


# ============================================================
# 25. Color vs Texture vs Combined 핵심 비교
# ============================================================

key_groups = [

    # --------------------------------------------------------
    # Lesion
    # --------------------------------------------------------

    "lesion_color_group",
    "lesion_texture_group",
    "lesion_color_texture_group",


    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    "delta_color_group",
    "delta_texture_group",
    "delta_color_texture_group",


    # --------------------------------------------------------
    # All
    # --------------------------------------------------------

    "all_color_group",
    "all_texture_group",
    "all_color_texture_group",
]


key_results = (
    results_df[
        results_df[
            "feature_group"
        ].isin(
            key_groups
        )
    ]
    .copy()
)


key_results = (
    key_results
    .sort_values(
        [
            "feature_group",
            "macro_f1_mean",
        ],
        ascending=[
            True,
            False,
        ]
    )
)


key_results_path = (
    OUT_DIR
    / "key_color_texture_comparison.csv"
)


key_results.to_csv(
    key_results_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)

print(
    "Key Color / Texture / Combined comparison"
)

print("=" * 70)


print(
    key_results[
        [
            "model",
            "feature_family",
            "feature_group",
            "n_samples",
            "n_features",
            "macro_f1_mean",
            "balanced_accuracy_mean",
            "accuracy_mean",
        ]
    ]
)


print(
    "\nSaved:",
    key_results_path
)


# ============================================================
# 26. Feature Group별 최고 모델 출력
# ============================================================

print()
print("=" * 70)

print(
    "Feature Group별 최고 성능 모델"
)

print("=" * 70)


best_by_group = (
    results_df
    .loc[
        results_df
        .groupby(
            "feature_group"
        )[
            "macro_f1_mean"
        ]
        .idxmax()
    ]
    .sort_values(
        "macro_f1_mean",
        ascending=False
    )
)


print(
    best_by_group[
        [
            "feature_family",
            "feature_group",
            "model",
            "n_samples",
            "n_features",
            "macro_f1_mean",
            "balanced_accuracy_mean",
            "accuracy_mean",
        ]
    ]
)


best_by_group_path = (
    OUT_DIR
    / "best_model_by_feature_group.csv"
)


best_by_group.to_csv(
    best_by_group_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 27. 전체 최고 결과
# ============================================================

print()
print("=" * 70)

print(
    "Overall Best Results"
)

print("=" * 70)


print(
    results_df[
        [
            "model",
            "feature_family",
            "feature_group",
            "n_samples",
            "n_features",
            "macro_f1_mean",
            "macro_f1_std",
            "balanced_accuracy_mean",
            "accuracy_mean",
        ]
    ]
    .head(
        20
    )
)


print()
print("=" * 70)

print(
    "ML analysis completed"
)

print("=" * 70)


print(
    "Output directory:",
    OUT_DIR
)