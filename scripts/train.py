#!/usr/bin/env python

import joblib
import pandas as pd
from pathlib import Path


from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
import xgboost as xgb




def load_data():
    data_url = 'data/fetal_health.csv'

    df = pd.read_csv(data_url)

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    return df




def train_model(df):
    features_list = ['prolongued_decelerations', 'accelerations', 'uterine_contractions', 'light_decelerations', 'fetal_movement', 'severe_decelerations', 'baseline_value']
    
    # Best XGB parameters: {'classifier__learning_rate': 0.05, 'classifier__max_depth': 6, 'classifier__n_estimators': 100}

    # y_train = df.fetal_health.values
    y_train = df.fetal_health.astype(int).to_numpy() - 1
    train_dict = df[features_list].to_dict(orient='records')

    pipeline = make_pipeline(
        DictVectorizer(),
        xgb.XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=100, random_state=42, eval_metric='logloss', n_jobs=-1)
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        train_dict,
        y_train,
        cv=cv,
        scoring={'roc_auc_ovr': 'roc_auc_ovr'},
    )
    print(f"ROC AUC OvR[one-vs-rest]: {scores['test_roc_auc_ovr'].mean():.3f} ")


    # Train the model to save, using all available training rows.
    pipeline.fit(train_dict, y_train)

    return pipeline


def save(pipeline):
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_pipeline.joblib"

    joblib.dump({"model": pipeline}, model_path)
    print(f"Successfully saved model to {model_path}")

df = load_data()
pipeline = train_model(df)
save(pipeline)
