import json
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from features import build_features

raw_df = pd.read_csv("ml/preauth_transactions_enriched.csv", low_memory=False)
raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
raw_df = raw_df.sort_values("timestamp").reset_index(drop=True)

split_idx = int(len(raw_df) * 0.8)
train_raw, val_raw = raw_df.iloc[:split_idx].copy(), raw_df.iloc[split_idx:].copy()

train_df, feature_cols, bin_map = build_features(train_raw, is_train=True)
val_df, _, _ = build_features(val_raw, is_train=False, bin_risk_map=bin_map)

X_train, y_train = train_df[feature_cols], train_df["is_chargeback"]
X_val, y_val = val_df[feature_cols], val_df["is_chargeback"]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
base_score = float(y_train.mean())

best_params = {
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "gamma": 1.0,
}

RUN_TUNING = True

if RUN_TUNING:
    print("Optuna...")
    def objective(trial):
        params = {
            "n_estimators": 300,
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.1, 3.0),
            "scale_pos_weight": scale_pos_weight,
            "base_score": base_score,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "early_stopping_rounds": 20,
            "random_state": 67,
            "n_jobs": -1,
        }
        m = xgb.XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = m.predict_proba(X_val)[:, 1]
        p, r, _ = precision_recall_curve(y_val, y_pred)
        return auc(r, p)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    print(f"best PR-AUC: {study.best_value:.4f}")

final_model = xgb.XGBClassifier(
    n_estimators=400,
    **best_params,
    scale_pos_weight=scale_pos_weight,
    base_score=base_score,
    objective="binary:logistic",
    eval_metric="aucpr",
    early_stopping_rounds=30,
    random_state=67,
)

final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50,
)

evals_result = final_model.evals_result()
history_df = pd.DataFrame(
    {
        "iteration": range(len(evals_result["validation_0"]["aucpr"])),
        "train_aucpr": evals_result["validation_0"]["aucpr"],
        "val_aucpr": evals_result["validation_1"]["aucpr"],
    }
)
history_df["gap"] = history_df["train_aucpr"] - history_df["val_aucpr"]
history_df.to_csv("model_artifacts/training_history.csv", index=False)

y_pred_proba = final_model.predict_proba(X_val)[:, 1]
precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
print(f"\nFinal PR-AUC: {auc(recall, precision):.4f}")
print(f"Final ROC-AUC: {roc_auc_score(y_val, y_pred_proba):.4f}")

schema = {"features": feature_cols, "bin_risk_map": bin_map}
with open("model_artifacts/features_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

final_model.save_model("model_artifacts/xgboost_baseline.json")
print("Final model and schema successfully saved to model_artifacts/")