import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import sklearn
from imblearn.over_sampling import SMOTE
from openai import OpenAI

# ChatGPT API 설정
client = OpenAI(api_key="OPENAI_API_KEY") # api키 추가해야 됨

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding
    
# MBTI 데이터셋 불러오기
mbti_data = pd.read_csv("MBTI 500.csv")

# 대화내용을 임베딩하여 데이터셋에 적용
mbti_data['embedding'] = mbti_data['posts'].apply(get_embedding)

# 데이터셋을 학습용과 테스트용으로 나눔
X_train, X_test, y_train, y_test = train_test_split(mbti_data['embedding'], mbti_data['type'], test_size=0.2, random_state=42)

# SMOTE 적용
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

# 임베딩 데이터를 numpy 배열로 변환
X_train_embed = np.array(X_train_over.tolist())
X_test_embed = np.array(X_test.tolist())

# 서포트 벡터 머신 분류 모델 학습
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_over, y_train_over)

# 테스트 데이터로 예측 => 여기에 사용자 텍스트를 임베딩하여 입력으로 추가
y_pred = svm_model.predict(X_test_embed)

# 성능 평가
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)
