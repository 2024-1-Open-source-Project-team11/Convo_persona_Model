import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import openai 
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

load_dotenv()
openai.api_key = os.getenv("API_KEY")

app = FastAPI()

class UserRequest(BaseModel):
    user_prompt_list : List[str]

class PredictionResult(BaseModel):
    mbti : str

@app.get("/")
def home():
    return "hello FastAPI Server"

@app.post("/mbti_prediction")
def predict_mbti(request : UserRequest):
    #한글로 받아온 user_prompt 영어로 번역
    user_prompt_list = request.user_prompt_list
    print(user_prompt_list)
    user_prompt_english = translate_to_english(user_prompt_list)
    print(user_prompt_english)  
    user_prompt_english_tfidf = tfidf.transform([user_prompt_english])
    combined_prediction = svm_model.predict(user_prompt_english_tfidf)  # 통합된 텍스트를 모델에 입력하여 예측 #정확도가 가장 높은 mbti를 결과로 반환
    print(combined_prediction)

    response = {
        'mbti' : combined_prediction[0]
    }

    logging.info(f"Model response : {response}")

    return response

# 영어로 번역하는 함수
def translate_to_english(text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content":f"Translate the following Korean text to English: '{text}'"}],
    )
    return completion.choices[0].message.content

recreate_model=False
if not os.path.isfile('tfidf_and_svm_model.pkl'):
    recreate_model=True

if recreate_model:    
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

    # 선형 커널 서포트 벡터 머신
    svm_model = LinearSVC(C=1.0)
    svm_model.fit(X_train_resampled, y_train_resampled)

    y_pred = svm_model.predict(X_test_tfidf)

    # 성능 평가
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    # 모델 저장
    joblib.dump((tfidf, svm_model), 'tfidf_and_svm_model.pkl')
else:
    tfidf, svm_model = joblib.load('tfidf_and_svm_model.pkl')

