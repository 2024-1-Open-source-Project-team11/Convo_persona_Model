import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

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

# Objective function for Random Forest
def objective_random_forest(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    model = RandomForestClassifier(**param)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study_random_forest = optuna.create_study(direction='maximize')
study_random_forest.optimize(objective_random_forest, n_trials=50)
print("Best Hyperparameters for Random Forest:", study_random_forest.best_params)

# 최적의 하이퍼파라미터로 모델 재학습 및 평가
best_params_rf = study_random_forest.best_params
model_rf = RandomForestClassifier(**best_params_rf)
model_rf.fit(X_train_resampled, y_train_resampled)
y_pred_rf = model_rf.predict(X_test_tfidf)
report_rf = classification_report(y_test, y_pred_rf)
print("Classification Report for Random Forest:")
print(report_rf)
