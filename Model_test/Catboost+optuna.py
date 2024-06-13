import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import catboost as cb

# MBTI 데이터셋 불러오기
mbti_data = pd.read_csv("MBTI 500.csv")

# 데이터셋을 학습용과 테스트용으로 나눔
X_train, X_test, y_train, y_test = train_test_split(mbti_data['posts'], mbti_data['type'], test_size=0.2, random_state=42)

# TF-IDF를 사용하여 텍스트를 벡터화
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Label Encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train_encoded)

def objective_catboost(trial):
    param = {
        'iterations': trial.suggest_int('iterations', 50, 200),
        'depth': trial.suggest_int('depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.01, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-9, 100, log=True),
    }
    model = cb.CatBoostClassifier(**param, verbose=0)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    return accuracy

# Optuna 스터디 생성 및 최적화
study_catboost = optuna.create_study(direction='maximize')
study_catboost.optimize(objective_catboost, n_trials=50)
print("Best Hyperparameters for CatBoost:", study_catboost.best_params)

# 최적의 하이퍼파라미터로 모델 재학습 및 평가
best_params_catboost = study_catboost.best_params
model_catboost = cb.CatBoostClassifier(**best_params_catboost, verbose=0)
model_catboost.fit(X_train_resampled, y_train_resampled)
y_pred_catboost = model_catboost.predict(X_test_tfidf)

# 성능 평가
report_catboost = classification_report(y_test_encoded, y_pred_catboost)
print("Classification Report for CatBoost:")
print(report_catboost)
