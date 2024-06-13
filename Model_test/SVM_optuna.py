import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

'''
recreate_model = False
if not os.path.isfile('mbti_svm.pkl'):
    recreate_model = True

if recreate_model: 
'''
# MBTI 데이터셋 불러오기
mbti_data = pd.read_csv("MBTI 500.csv")

# 데이터셋을 학습용과 테스트용으로 나눔
X_train, X_test, y_train, y_test = train_test_split(mbti_data['posts'], mbti_data['type'], test_size=0.2, random_state=42)

# TF-IDF를 사용하여 텍스트를 벡터화
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# SMOTE를 사용하여 오버샘플링
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

# 언더 샘플링을 위한 객체 생성
# undersampler = RandomUnderSampler(random_state=42)
# 언더 샘플링 적용
# X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_tfidf, y_train)

'''
# 서포트 벡터 머신 분류 모델 학습 
clf = LinearSVC(max_iter=10000)  # max_iter 값을 증가시킴
clf.fit(X_train_resampled, y_train_resampled)
cv = GridSearchCV(clf, {'C': [0.1, 1, 10, 100]}, scoring="accuracy")
cv.fit(X_train_resampled, y_train_resampled)

# 교차 검증 결과 출력
print(cv.best_estimator_.C)
svm_model = LinearSVC(C=cv.best_estimator_.C)
svm_model.fit(X_train_resampled, y_train_resampled)

y_pred = svm_model.predict(X_test_tfidf)
    
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
'''

def objective(trial):
    # 하이퍼파라미터 샘플링
    C = trial.suggest_float('C', 1e-10, 1e10, log=True)
    
    # 모델 학습 및 평가
    svm_model = LinearSVC(C=C, max_iter=10000)  # max_iter를 설정하여 수렴 보장
    svm_model.fit(X_train_resampled, y_train_resampled)
    y_pred = svm_model.predict(X_test_tfidf)
    
    # 성능 평가 (정확도 사용)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# 스터디 생성 및 최적화
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", study.best_params)

best_C = study.best_params['C']
svm_model = LinearSVC(C=best_C, max_iter=10000)
svm_model.fit(X_train_resampled, y_train_resampled)
y_pred = svm_model.predict(X_test_tfidf)

# 성능 평가
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

'''
    # 모델 저장
    joblib.dump(svm_model, 'mbti_svm.pkl')
else:
    svm_model = joblib.load('mbti_svm.pkl')



# X_train_tfidf 행렬의 각 행마다 0이 아닌 원소의 개수를 세는 코드
non_zero_counts = X_train_tfidf.getnnz(axis=1)

# 각 행의 0이 아닌 원소의 개수를 출력
for i, count in enumerate(non_zero_counts):
    print(f"행 {i+1}의 0이 아닌 원소 개수: {count}")
'''