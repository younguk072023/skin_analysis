from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)
from models.model_zoo import get_models
from sklearn.base import clone
from features.feature_groups import feature_groups

ROOT = Path(r"C:\Users\park_younguk\Desktop\skin")
CSV_PATH = ROOT / "lesion_background_ita_lab_features.csv"
OUT_DIR = ROOT / "ml_results"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

severity_order = ["mild", "moderate", "severe", "very severe"]
df = df[df["severity"].isin(severity_order)].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["severity"])

print("Class mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(i, "->", cls)

print("\nClass counts:")
print(df["severity"].value_counts().reindex(severity_order))

required_features = sorted({
    feature
    for feature_cols in feature_groups.values()
    for feature in feature_cols
})

missing_features = [
    feature
    for feature in required_features
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "CSV에 다음 feature가 없습니다:\n"
        + "\n".join(missing_features)
    )

print("\nNumber of feature groups:", len(feature_groups))
print("Number of unique features:", len(required_features))

models = get_models(random_state=42)

def evaluate_group(model_name, model, group_name, feature_cols):
    print("\n" + "=" * 60)
    print("Model:", model_name)
    print("Feature group:", group_name)
    print("Features:", feature_cols)

    model_out_dir = OUT_DIR / model_name
    model_out_dir.mkdir(exist_ok=True)

    data = df[feature_cols + ["severity"]].dropna().copy()

    class_counts = (
    data["severity"]
    .value_counts()
    .reindex(severity_order, fill_value=0)
    )

    if (class_counts < 5).any():
        raise ValueError(
            f"{model_name} / {group_name}: "
            f"5-fold CV를 수행하기에는 일부 클래스 표본이 부족합니다.\n"
            f"{class_counts}"
        )

    X = data[feature_cols]
    y_group = label_encoder.transform(data["severity"])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accs = []
    macro_f1s = []
    bal_accs = []

    y_true_all = []
    y_pred_all = []
    importance_list = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_group), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_group[train_idx]
        y_test = y_group[test_idx]

        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        accs.append(acc)
        macro_f1s.append(macro_f1)
        bal_accs.append(bal_acc)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

        if hasattr(clf, "feature_importances_"):
            importance_list.append(clf.feature_importances_)

        print(
            f"Fold {fold}: "
            f"Acc={acc:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Acc={bal_acc:.4f}"
        )

    print("\n5-fold CV results:")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    print(f"Balanced Accuracy: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")

    print("\nClassification report from all folds:")
    print(classification_report(
        y_true_all,
        y_pred_all,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=np.arange(len(label_encoder.classes_))
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} - {group_name}")
    plt.tight_layout()


    model_out_dir = OUT_DIR / model_name
    model_out_dir.mkdir(exist_ok=True)
    cm_path = model_out_dir / f"confusion_matrix_{group_name}.png"

    plt.savefig(cm_path, dpi=300)
    plt.close()

    if len(importance_list) > 0:
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance_mean": np.mean(importance_list, axis=0),
            "importance_std": np.std(importance_list, axis=0)
        }).sort_values("importance_mean", ascending=False)

        importance_path = model_out_dir / f"feature_importance_{group_name}.csv"
        importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")

    return {
        "model": model_name,
        "feature_group": group_name,
        "n_samples": len(data),
        "n_features": len(feature_cols),

        "accuracy_mean": np.mean(accs),
        "accuracy_std": np.std(accs),

        "macro_f1_mean": np.mean(macro_f1s),
        "macro_f1_std": np.std(macro_f1s),

        "balanced_accuracy_mean": np.mean(bal_accs),
        "balanced_accuracy_std": np.std(bal_accs),
    }

results = []

for model_name, model in models.items():
    for group_name, feature_cols in feature_groups.items():
        if len(feature_cols) == 0:
            continue

        result = evaluate_group(model_name, model, group_name, feature_cols)
        results.append(result)

results_df = pd.DataFrame(results)
results_path = OUT_DIR / "ml_group_comparison_results.csv"
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")

for model_name in results_df["model"].unique():
    model_df = results_df[results_df["model"] == model_name].copy()
    model_path = OUT_DIR / model_name / "group_comparison_results.csv"
    model_df.to_csv(model_path, index=False, encoding="utf-8-sig")

print("\nSaved group comparison:", results_path)
print(results_df.sort_values("macro_f1_mean", ascending=False))

# ==============================================================================
# 특징 그룹(Feature Group) 중심 피처-모델 비교 분석 및 저장
# ==============================================================================
print("\n" + "=" * 60)
print("특징 그룹(Feature Group) 중심 결과 재정리 중...")

# 1. Feature Group을 기준으로 Pivot Table 생성 (Macro F1-score 기준)
pivot_f1_mean = results_df.pivot(
    index="feature_group", 
    columns="model", 
    values="macro_f1_mean"
)
pivot_f1_std = results_df.pivot(
    index="feature_group", 
    columns="model", 
    values="macro_f1_std"
)

# 2. Accuracy 기준 Pivot Table도 생성
pivot_acc_mean = results_df.pivot(
    index="feature_group", 
    columns="model", 
    values="accuracy_mean"
)

# 3. 평균 ± 표준편차 형태의 가독성 좋은 종합 비교 표 생성 (예: 0.8524 ± 0.0120)
formatted_summary = pd.DataFrame(index=pivot_f1_mean.index)

for col in pivot_f1_mean.columns:
    formatted_summary[f"{col} (F1)"] = (
        pivot_f1_mean[col].map("{:.4f}".format) + " ± " + pivot_f1_std[col].map("{:.4f}".format)
    )

# 최고 성능 모델 및 해당 점수 열 추가
formatted_summary["Best_Model"] = pivot_f1_mean.idxmax(axis=1)
formatted_summary["Best_Macro_F1"] = pivot_f1_mean.max(axis=1).map("{:.4f}".format)

# 4. CSV 파일로 저장
feature_centric_path = OUT_DIR / "feature_centric_model_comparison.csv"
formatted_summary.to_csv(feature_centric_path, encoding="utf-8-sig")
print("Saved feature-centric comparison table:", feature_centric_path)

# 5. 특징 그룹 중심 성능 비교 히트맵(Heatmap) 시각화 저장
plt.figure(figsize=(10, 6))
sns.heatmap(
    pivot_f1_mean, 
    annot=True, 
    fmt=".4f", 
    cmap="YlGnBu", 
    cbar_kws={'label': 'Mean Macro F1-score'}
)
plt.title("Macro F1-score by Feature Group & Model", fontsize=14, pad=15)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Feature Group", fontsize=12)
plt.tight_layout()

heatmap_path = OUT_DIR / "feature_centric_f1_heatmap.png"
plt.savefig(heatmap_path, dpi=300)
plt.close()
print("Saved feature-centric heatmap:", heatmap_path)

# 콘솔에 출력해서 바로 확인하기
print("\n[ 특징 그룹별 최고 성능 모델 요약 ]")
print(formatted_summary[["Best_Model", "Best_Macro_F1"]])