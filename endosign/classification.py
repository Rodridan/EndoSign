"""
Classification models for Enzian staging and endometriosis.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_classifier(model_name='random_forest', random_state=42):
    """
    Returns a classifier instance by model name.
    Supported: 'random_forest', 'logistic_regression', 'svc', 'xgboost'
    """
    if model_name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, random_state=random_state)
    elif model_name == 'svc':
        return SVC(probability=True, random_state=random_state)
    elif model_name == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def train_and_evaluate_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model_name='random_forest',
    test_size=0.2,
    random_state=42
):
    """
    Trains and evaluates a classifier.
    Returns: fitted model, test report dict.
    """
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    clf = get_classifier(model_name, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return clf, report

def cross_validate_any_classifier(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model_name='random_forest',
    n_folds=5,
    random_state=42
):
    """
    Perform k-fold cross-validation for the selected classifier.
    Returns mean and std of accuracy.
    """
    X = df[feature_cols]
    y = df[target_col]
    clf = get_classifier(model_name, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=n_folds)
    return scores.mean(), scores.std()
