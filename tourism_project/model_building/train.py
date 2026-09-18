# for data manipulation
import os
import pandas as pd
# for building the preprocessing and modeling pipeline
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for model serialization and experiment tracking
import joblib
import mlflow

# The GitHub Actions workflow starts a local `mlflow ui` server (backed by the
# free, file-based tracking store) on the runner before this script executes,
# so tracking works without any paid/hosted MLflow service. The URI can be
# overridden with an environment variable for local experimentation.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.set_experiment("Tourism_Wellness_Package_Prediction")

# Xtrain/Xtest/ytrain/ytest are downloaded from the previous job's artifact
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

# Set the class weight to handle class imbalance (computed from the training
# split only, so the test set never influences this value)
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42, eval_metric="logloss")

# Define hyperparameter grid.
# Kept intentionally small so GridSearchCV (cv=5) completes quickly inside a
# GitHub Actions runner, while still covering the parameters that most affect
# XGBoost performance: tree count/depth, feature sampling, learning rate and
# L2 regularization.
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__colsample_bytree': [0.8],
    'xgbclassifier__colsample_bylevel': [0.8],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__reg_lambda': [1, 5],
}
# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV.
    # scoring="f1" is used (rather than plain accuracy) because ProdTaken is
    # imbalanced (~81% / 19%): a model that always predicts "no purchase"
    # would score ~81% accuracy while being useless for the business, whereas
    # F1 balances precision and recall on the minority (purchasing) class,
    # which is the one the business actually needs to identify correctly.
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination tried during the search as a nested run,
    # so all experiments can be compared side by side in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_test_f1", results["std_test_score"][i])

    # Log the best hyperparameters in the main run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Classification threshold: 0.45 (slightly below the default 0.5) is used
    # to trade a little precision for extra recall, since missing a genuine
    # buyer (false negative) costs the business a lost sale, while contacting
    # a customer who does not buy (false positive) is comparatively cheap.
    classification_threshold = 0.45

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })

    print("Best hyperparameters:", grid_search.best_params_)
    print("Best CV F1 score:", grid_search.best_score_)
    print("\nTest set classification report:")
    print(classification_report(ytest, y_pred_test))
    print("Test accuracy:", test_report["accuracy"])
    test_roc_auc = roc_auc_score(ytest, y_pred_test_proba)
    print("Test ROC-AUC:", test_roc_auc)
    mlflow.log_metric("test_roc_auc", test_roc_auc)

    # Save the model next to app.py so the Streamlit app can load it directly,
    # and log it as an MLflow artifact for traceability
    model_path = "tourism_project/deployment/best_tourism_package_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
