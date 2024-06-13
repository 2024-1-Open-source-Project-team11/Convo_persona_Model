import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

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

def objective_xgboost(trial):
    param = {
        'verbosity': 0,
        'objective': 'multi:softmax',
        'num_class': 16,  # Number of MBTI classes
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**param)
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    return accuracy

study_xgboost = optuna.create_study(direction='maximize')
study_xgboost.optimize(objective_xgboost, n_trials=50)
print("Best Hyperparameters for XGBoost:", study_xgboost.best_params)

# 최적의 하이퍼파라미터로 모델 재학습 및 평가
best_params_xgboost = study_xgboost.best_params
model_xgboost = xgb.XGBClassifier(**best_params_xgboost)
model_xgboost.fit(X_train_resampled, y_train_resampled)
y_pred_xgboost = model_xgboost.predict(X_test_tfidf)

# 성능 평가
report_xgboost = classification_report(y_test_encoded, y_pred_xgboost)
print("Classification Report for XGBoost:")
print(report_xgboost)
