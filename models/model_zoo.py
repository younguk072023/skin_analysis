from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier 

def get_models(random_state=42):

    models = {

        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="linear",
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

        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            max_depth=None,
            min_samples_leaf=2
            ))
        ]),

        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced",
            min_samples_leaf=2
            ))
        ]),

        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                random_state=random_state
            ))
        ]),

        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=500,
                random_state=random_state,
                eval_metric="logloss" 
            ))
        ])

    }

    return models