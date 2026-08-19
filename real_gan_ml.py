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
    confusion_matrix,
)
from sklearn.base import clone

from models.model_zoo import get_models

from features.feature_color import feature_color
from features.feature_texture import feature_texture
from features.feature_combined import feature_combined


# ============================================================
# 1. PATH
# ============================================================

ROOT = Path(
    r"C:\Users\park_younguk\Desktop\skin"
)


# Real feature
REAL_CSV = (
    ROOT
    / "lesion_background_color_texture_features.csv"
)


# GAN feature
GAN_CSV = Path(
    r"E:\gan_analysis_001612"
    r"\gan_lesion_background_color_texture_features.csv"
)


OUT_DIR = (
    ROOT
    / "ml_results_real_vs_gan_augmentation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 설정
# ============================================================

N_SPLITS = 5

RANDOM_STATE = 42


severity_order = [
    "mild",
    "moderate",
    "severe",
    "very severe",
]


# ============================================================
# 3. Severity 이름 통일
#
# very_severe
# very severe
#
# 둘 다 "very severe"로 통일
# ============================================================

def normalize_severity(value):

    value = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
    )

    value = " ".join(
        value.split()
    )

    return value


# ============================================================
# 4. CSV Load
# ============================================================

real_df = pd.read_csv(
    REAL_CSV
)

gan_df = pd.read_csv(
    GAN_CSV
)


real_df["severity"] = (
    real_df["severity"]
    .apply(normalize_severity)
)

gan_df["severity"] = (
    gan_df["severity"]
    .apply(normalize_severity)
)


# 유효 severity만 사용
real_df = real_df[
    real_df["severity"].isin(
        severity_order
    )
].copy()


gan_df = gan_df[
    gan_df["severity"].isin(
        severity_order
    )
].copy()


# ============================================================
# 5. Dataset 정보
# ============================================================

print("=" * 70)
print("REAL DATA")
print("=" * 70)

print(
    real_df["severity"]
    .value_counts()
    .reindex(
        severity_order
    )
)


print()

print("=" * 70)
print("GAN DATA")
print("=" * 70)

print(
    gan_df["severity"]
    .value_counts()
    .reindex(
        severity_order
    )
)


# ============================================================
# 6. Label encoder
# ============================================================

label_encoder = LabelEncoder()

label_encoder.fit(
    severity_order
)


print()
print("Class mapping:")

for i, cls in enumerate(
    label_encoder.classes_
):

    print(
        i,
        "->",
        cls
    )


# ============================================================
# 7. Feature Groups
# ============================================================

feature_groups = {
    **feature_color,
    **feature_texture,
    **feature_combined,
}


feature_family_map = {}


for name in feature_color:

    feature_family_map[name] = (
        "color"
    )


for name in feature_texture:

    feature_family_map[name] = (
        "texture"
    )


for name in feature_combined:

    feature_family_map[name] = (
        "combined"
    )


# ============================================================
# 8. Feature 존재 여부
# ============================================================

required_features = sorted({

    feature

    for cols in feature_groups.values()

    for feature in cols
})


missing_real = [

    feature

    for feature
    in required_features

    if feature
    not in real_df.columns
]


missing_gan = [

    feature

    for feature
    in required_features

    if feature
    not in gan_df.columns
]


if missing_real:

    raise ValueError(
        "Real CSV missing features:\n"
        + "\n".join(
            missing_real
        )
    )


if missing_gan:

    raise ValueError(
        "GAN CSV missing features:\n"
        + "\n".join(
            missing_gan
        )
    )


print()
print("=" * 70)
print("FEATURE CHECK")
print("=" * 70)

print(
    "Feature groups:",
    len(feature_groups)
)

print(
    "Unique features:",
    len(required_features)
)


# ============================================================
# 9. Models
# ============================================================

models = get_models(
    random_state=
        RANDOM_STATE
)


print()
print("Models:")

for model_name in models:

    print(
        "-",
        model_name
    )


# ============================================================
# 10. GAN Sampling
#
# Real Train의 severity별 N만큼
# GAN을 sampling
#
# 즉
# Real : GAN = 1 : 1
# ============================================================

def sample_gan_for_fold(
    gan_data,
    real_train,
    fold
):

    sampled_parts = []


    for class_index, severity in enumerate(
        severity_order
    ):

        # ----------------------------------------------------
        # Real train의 해당 class N
        # ----------------------------------------------------

        n_real = int(
            (
                real_train["severity"]
                == severity
            ).sum()
        )


        # ----------------------------------------------------
        # GAN pool
        # ----------------------------------------------------

        pool = gan_data[
            gan_data["severity"]
            == severity
        ]


        if len(pool) == 0:

            raise ValueError(
                f"GAN 데이터에 "
                f"{severity} class가 없습니다."
            )


        # ----------------------------------------------------
        # GAN 데이터가 충분하면
        # Real train과 동일한 개수
        #
        # 부족하면 가능한 만큼 사용
        # ----------------------------------------------------

        n_sample = min(
            n_real,
            len(pool)
        )


        if n_sample < n_real:

            print(
                f"[WARNING] "
                f"{severity}: "
                f"Real train={n_real}, "
                f"GAN available={len(pool)}"
            )


        # ----------------------------------------------------
        # 고정 random seed
        #
        # 모델이 달라져도
        # 같은 fold / severity에서는
        # 같은 GAN sample이 선택되도록 함
        # ----------------------------------------------------

        random_seed = (
            RANDOM_STATE
            + fold * 100
            + class_index
        )


        sampled = (
            pool.sample(
                n=n_sample,
                replace=False,
                random_state=
                    random_seed
            )
            .copy()
        )


        sampled_parts.append(
            sampled
        )


    return pd.concat(
        sampled_parts,
        ignore_index=True
    )


# ============================================================
# 11. Fold별 평가
# ============================================================

def evaluate_group(
    model_name,
    model,
    group_name,
    feature_cols
):

    print()
    print("=" * 75)

    print(
        "Model:",
        model_name
    )

    print(
        "Feature group:",
        group_name
    )

    print(
        "Features:",
        len(feature_cols)
    )

    print("=" * 75)


    # ========================================================
    # Real / GAN
    #
    # 이 feature group에 필요한 데이터만 사용
    # ========================================================

    real_data = (
        real_df[
            feature_cols
            + ["severity"]
        ]
        .dropna()
        .reset_index(
            drop=True
        )
    )


    gan_data = (
        gan_df[
            feature_cols
            + ["severity"]
        ]
        .dropna()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Real class counts
    # ========================================================

    real_counts = (
        real_data["severity"]
        .value_counts()
        .reindex(
            severity_order,
            fill_value=0
        )
    )


    if (
        real_counts < N_SPLITS
    ).any():

        raise ValueError(
            f"{group_name}: "
            f"Real 데이터가 5-fold에 부족합니다.\n"
            f"{real_counts}"
        )


    # ========================================================
    # Real X / y
    # ========================================================

    X_real = (
        real_data[
            feature_cols
        ]
    )


    y_real = (
        label_encoder
        .transform(
            real_data[
                "severity"
            ]
        )
    )


    # ========================================================
    # 같은 Real fold를
    #
    # Real only
    # Real + GAN
    #
    # 둘 다 사용
    # ========================================================

    cv = StratifiedKFold(
        n_splits=
            N_SPLITS,
        shuffle=True,
        random_state=
            RANDOM_STATE
    )


    # ========================================================
    # 결과 저장
    # ========================================================

    baseline_metrics = []

    augmented_metrics = []


    baseline_true = []
    baseline_pred = []

    augmented_true = []
    augmented_pred = []


    fold_rows = []


    # ========================================================
    # Fold
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(

        cv.split(
            X_real,
            y_real
        ),

        start=1
    ):


        # ====================================================
        # Real Train / Test
        # ====================================================

        real_train = (
            real_data.iloc[
                train_idx
            ]
            .copy()
        )


        real_test = (
            real_data.iloc[
                test_idx
            ]
            .copy()
        )


        X_train_real = (
            real_train[
                feature_cols
            ]
        )


        y_train_real = (
            label_encoder
            .transform(
                real_train[
                    "severity"
                ]
            )
        )


        X_test = (
            real_test[
                feature_cols
            ]
        )


        y_test = (
            label_encoder
            .transform(
                real_test[
                    "severity"
                ]
            )
        )


        # ====================================================
        # A. BASELINE
        #
        # Real train only
        # ====================================================

        baseline_clf = clone(
            model
        )


        baseline_clf.fit(
            X_train_real,
            y_train_real
        )


        pred_baseline = (
            baseline_clf.predict(
                X_test
            )
        )


        baseline_acc = (
            accuracy_score(
                y_test,
                pred_baseline
            )
        )


        baseline_f1 = (
            f1_score(
                y_test,
                pred_baseline,
                average="macro",
                zero_division=0
            )
        )


        baseline_bal = (
            balanced_accuracy_score(
                y_test,
                pred_baseline
            )
        )


        baseline_metrics.append(
            (
                baseline_acc,
                baseline_f1,
                baseline_bal
            )
        )


        baseline_true.extend(
            y_test
        )

        baseline_pred.extend(
            pred_baseline
        )


        # ====================================================
        # B. GAN AUGMENTATION
        #
        # GAN sample:
        # 각 class별 Real Train과 동일한 N
        # ====================================================

        gan_train = (
            sample_gan_for_fold(
                gan_data=
                    gan_data,

                real_train=
                    real_train,

                fold=
                    fold
            )
        )


        # ----------------------------------------------------
        # Real + GAN
        # ----------------------------------------------------

        augmented_train = (
            pd.concat(
                [
                    real_train,
                    gan_train
                ],
                ignore_index=True
            )
        )


        X_train_aug = (
            augmented_train[
                feature_cols
            ]
        )


        y_train_aug = (
            label_encoder
            .transform(
                augmented_train[
                    "severity"
                ]
            )
        )


        # ----------------------------------------------------
        # 새 모델
        # ----------------------------------------------------

        augmented_clf = clone(
            model
        )


        augmented_clf.fit(
            X_train_aug,
            y_train_aug
        )


        # ----------------------------------------------------
        # Test는 반드시
        # 같은 REAL TEST
        # ----------------------------------------------------

        pred_augmented = (
            augmented_clf.predict(
                X_test
            )
        )


        augmented_acc = (
            accuracy_score(
                y_test,
                pred_augmented
            )
        )


        augmented_f1 = (
            f1_score(
                y_test,
                pred_augmented,
                average="macro",
                zero_division=0
            )
        )


        augmented_bal = (
            balanced_accuracy_score(
                y_test,
                pred_augmented
            )
        )


        augmented_metrics.append(
            (
                augmented_acc,
                augmented_f1,
                augmented_bal
            )
        )


        augmented_true.extend(
            y_test
        )

        augmented_pred.extend(
            pred_augmented
        )


        # ====================================================
        # Fold 결과
        # ====================================================

        fold_rows.append({

            "model":
                model_name,

            "feature_family":
                feature_family_map[
                    group_name
                ],

            "feature_group":
                group_name,

            "fold":
                fold,

            "n_real_train":
                len(
                    real_train
                ),

            "n_gan_train":
                len(
                    gan_train
                ),

            "n_real_test":
                len(
                    real_test
                ),

            "real_only_accuracy":
                baseline_acc,

            "real_plus_gan_accuracy":
                augmented_acc,

            "delta_accuracy":
                augmented_acc
                - baseline_acc,

            "real_only_macro_f1":
                baseline_f1,

            "real_plus_gan_macro_f1":
                augmented_f1,

            "delta_macro_f1":
                augmented_f1
                - baseline_f1,

            "real_only_balanced_accuracy":
                baseline_bal,

            "real_plus_gan_balanced_accuracy":
                augmented_bal,

            "delta_balanced_accuracy":
                augmented_bal
                - baseline_bal,
        })


        print(
            f"Fold {fold} | "
            f"Real F1={baseline_f1:.4f} | "
            f"Real+GAN F1={augmented_f1:.4f} | "
            f"Δ={augmented_f1-baseline_f1:+.4f}"
        )


    # ========================================================
    # Numpy
    # ========================================================

    baseline_metrics = np.array(
        baseline_metrics
    )


    augmented_metrics = np.array(
        augmented_metrics
    )


    # ========================================================
    # 평균
    # ========================================================

    result = {

        "model":
            model_name,

        "feature_family":
            feature_family_map[
                group_name
            ],

        "feature_group":
            group_name,

        "n_features":
            len(
                feature_cols
            ),

        "n_real":
            len(
                real_data
            ),

        "n_gan_available":
            len(
                gan_data
            ),


        # ----------------------------------------------------
        # Real only
        # ----------------------------------------------------

        "real_accuracy_mean":
            baseline_metrics[
                :, 0
            ].mean(),

        "real_accuracy_std":
            baseline_metrics[
                :, 0
            ].std(),

        "real_macro_f1_mean":
            baseline_metrics[
                :, 1
            ].mean(),

        "real_macro_f1_std":
            baseline_metrics[
                :, 1
            ].std(),

        "real_balanced_accuracy_mean":
            baseline_metrics[
                :, 2
            ].mean(),

        "real_balanced_accuracy_std":
            baseline_metrics[
                :, 2
            ].std(),


        # ----------------------------------------------------
        # Real + GAN
        # ----------------------------------------------------

        "aug_accuracy_mean":
            augmented_metrics[
                :, 0
            ].mean(),

        "aug_accuracy_std":
            augmented_metrics[
                :, 0
            ].std(),

        "aug_macro_f1_mean":
            augmented_metrics[
                :, 1
            ].mean(),

        "aug_macro_f1_std":
            augmented_metrics[
                :, 1
            ].std(),

        "aug_balanced_accuracy_mean":
            augmented_metrics[
                :, 2
            ].mean(),

        "aug_balanced_accuracy_std":
            augmented_metrics[
                :, 2
            ].std(),
    }


    # ========================================================
    # Delta
    # ========================================================

    result[
        "delta_accuracy"
    ] = (
        result[
            "aug_accuracy_mean"
        ]
        -
        result[
            "real_accuracy_mean"
        ]
    )


    result[
        "delta_macro_f1"
    ] = (
        result[
            "aug_macro_f1_mean"
        ]
        -
        result[
            "real_macro_f1_mean"
        ]
    )


    result[
        "delta_balanced_accuracy"
    ] = (
        result[
            "aug_balanced_accuracy_mean"
        ]
        -
        result[
            "real_balanced_accuracy_mean"
        ]
    )


    # ========================================================
    # Confusion Matrix 저장
    # ========================================================

    model_dir = (
        OUT_DIR
        / model_name
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Real only
    # --------------------------------------------------------

    cm_real = confusion_matrix(
        baseline_true,
        baseline_pred,
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
        cm_real,
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
        f"{model_name}\n"
        f"{group_name}\n"
        f"Real Only"
    )


    plt.tight_layout()


    plt.savefig(

        model_dir
        / f"{group_name}_real_only_cm.png",

        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # --------------------------------------------------------
    # Real + GAN
    # --------------------------------------------------------

    cm_aug = confusion_matrix(
        augmented_true,
        augmented_pred,
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
        cm_aug,
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
        f"{model_name}\n"
        f"{group_name}\n"
        f"Real + GAN"
    )


    plt.tight_layout()


    plt.savefig(

        model_dir
        / f"{group_name}_real_plus_gan_cm.png",

        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    return (
        result,
        fold_rows
    )


# ============================================================
# 12. ALL MODEL × FEATURE GROUP
# ============================================================

results = []

all_fold_rows = []


for (
    model_name,
    model
) in models.items():


    for (
        group_name,
        feature_cols
    ) in feature_groups.items():


        if len(
            feature_cols
        ) == 0:

            continue


        (
            result,
            fold_rows
        ) = evaluate_group(

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


        all_fold_rows.extend(
            fold_rows
        )


# ============================================================
# 13. 결과 저장
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        "delta_macro_f1",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


fold_df = pd.DataFrame(
    all_fold_rows
)


RESULT_PATH = (
    OUT_DIR
    / "real_vs_real_plus_gan_results.csv"
)


FOLD_PATH = (
    OUT_DIR
    / "real_vs_real_plus_gan_fold_results.csv"
)


results_df.to_csv(
    RESULT_PATH,
    index=False,
    encoding="utf-8-sig"
)


fold_df.to_csv(
    FOLD_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 핵심 Feature Group만 저장
# ============================================================

key_groups = [

    "lesion_color_group",

    "lesion_texture_group",

    "lesion_color_texture_group",

    "delta_color_group",

    "delta_texture_group",

    "delta_color_texture_group",

    "all_color_group",

    "all_texture_group",

    "all_color_texture_group",
]


key_df = (
    results_df[
        results_df[
            "feature_group"
        ].isin(
            key_groups
        )
    ]
    .copy()
)


KEY_PATH = (
    OUT_DIR
    / "key_real_vs_gan_comparison.csv"
)


key_df.to_csv(
    KEY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 15. 결과 출력
# ============================================================

print()
print("=" * 80)
print("REAL ONLY vs REAL + GAN")
print("=" * 80)


print(
    results_df[
        [
            "model",
            "feature_family",
            "feature_group",

            "real_macro_f1_mean",
            "aug_macro_f1_mean",
            "delta_macro_f1",

            "real_balanced_accuracy_mean",
            "aug_balanced_accuracy_mean",
            "delta_balanced_accuracy",

            "real_accuracy_mean",
            "aug_accuracy_mean",
            "delta_accuracy",
        ]
    ]
    .head(
        30
    )
)


print()
print("=" * 80)
print("BEST GAN IMPROVEMENTS")
print("=" * 80)


print(
    results_df[
        [
            "model",
            "feature_group",
            "real_macro_f1_mean",
            "aug_macro_f1_mean",
            "delta_macro_f1",
        ]
    ]
    .head(
        20
    )
)


print()
print("Saved:")
print(RESULT_PATH)
print(FOLD_PATH)
print(KEY_PATH)
print()
print("ML augmentation analysis completed.")