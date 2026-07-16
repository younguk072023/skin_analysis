from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_models(random_state=42):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state
        ),

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ))
        ]),

        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=random_state
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    return models