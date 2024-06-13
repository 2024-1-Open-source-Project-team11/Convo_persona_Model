import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# MBTI 데이터셋 불러오기
mbti_data = pd.read_csv("MBTI 500.csv")

# 데이터셋을 학습용과 테스트용으로 나눔
X_train, X_test, y_train, y_test = train_test_split(mbti_data['posts'], mbti_data['type'], test_size=0.2, random_state=42)

# TF-IDF를 사용하여 텍스트를 벡터화
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

def objective_lightgbm(trial):
    param = {
        'objective': 'multiclass',
        'num_class': 16,  # Assuming 16 MBTI types
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = lgb.LGBMClassifier(**param)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study_lightgbm = optuna.create_study(direction='maximize')
study_lightgbm.optimize(objective_lightgbm, n_trials=50)
print("Best Hyperparameters for LightGBM:", study_lightgbm.best_params)

# 최적의 하이퍼파라미터로 모델 재학습 및 평가
best_params_lightgbm = study_lightgbm.best_params
model_lightgbm = lgb.LGBMClassifier(**best_params_lightgbm)
model_lightgbm.fit(X_train_resampled, y_train_resampled)
y_pred_lightgbm = model_lightgbm.predict(X_test_tfidf)
report_lightgbm = classification_report(y_test, y_pred_lightgbm)
print("Classification Report for LightGBM:")
print(report_lightgbm)
