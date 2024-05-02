import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle
import joblib
import os

recreate_model=False
if not os.path.isfile('mbti_svm.pkl'):
    recreate_model=True

if recreate_model:    
   # MBTI 데이터셋 불러오기
    mbti_data = pd.read_csv("MBTI 500.csv")

    # 벡터화
    tfidf = TfidfVectorizer()

    # 데이터셋을 학습용과 테스트용으로 나눔
    X_train, X_test, y_train, y_test = train_test_split(mbti_data['posts'], mbti_data['type'], test_size=0.2, random_state=42)
    X_train_tfidf = tfidf.fit_transform(X_train)

    # 서포트 벡터 머신 분류 모델 학습
    clf = LinearSVC(C=0.4)
    clf.fit(X_train_tfidf, y_train)

    # 그리드 서치 모델 인스턴스화
    svm_model = Pipeline([('tfidf',TfidfVectorizer()),('clf',clf)])
    svm_model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(svm_model, 'mbti_svm.pkl')
else:
    svm_model = joblib.load('mbti_svm.pkl')

recreate_user_embeddings = False
if not os.path.isfile('user_embeddings.pkl'):
    recreate_user_embeddings = True

if recreate_user_embeddings:    
    user_embeddings = []
else:
    with open('user_embeddings.pkl', 'rb') as file:
        user_embeddings = pickle.load(file)

user_text = input()
user_embeddings.append(user_text)
combined_text = ' '.join(user_embeddings)  # 두 개의 텍스트를 공백을 이용하여 통합
combined_prediction = svm_model.predict([combined_text])  # 통합된 텍스트를 모델에 입력하여 예측
print(combined_prediction)
print(user_embeddings)
print(combined_text)

with open('user_embeddings.pkl', 'wb') as file:
    pickle.dump(user_embeddings, file)