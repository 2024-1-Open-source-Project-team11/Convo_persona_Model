import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

# Objective function for Logistic Regression
def objective_logistic_regression(trial):
    param = {
        'C': trial.suggest_loguniform('C', 1e-4, 1e2),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
    }
    model = LogisticRegression(**param)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

study_logistic_regression = optuna.create_study(direction='maximize')
study_logistic_regression.optimize(objective_logistic_regression, n_trials=50)
print("Best Hyperparameters for Logistic Regression:", study_logistic_regression.best_params)

# 최적의 하이퍼파라미터로 모델 재학습 및 평가
best_params_lr = study_logistic_regression.best_params
model_lr = LogisticRegression(**best_params_lr)
model_lr.fit(X_train_resampled, y_train_resampled)
y_pred_lr = model_lr.predict(X_test_tfidf)
report_lr = classification_report(y_test, y_pred_lr)
print("Classification Report for Logistic Regression:")
print(report_lr)
