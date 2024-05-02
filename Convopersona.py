import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pickle
import joblib
import os
from fastapi import FastAPI
import openai 

def get_api_key():
    # 환경 변수에서 API 키 가져오기
    api_key = os.environ.get("OPENAI_API")
    if not api_key:
        raise ValueError("API 키를 찾을 수 없습니다. 환경 변수를 설정하세요.")
    return api_key
openai.api_key = get_api_key()

app = FastAPI()

# 영어로 번역하는 함수
def translate_to_english(text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content":f"Translate the following Korean text to English: '{text}'"}],
    )
    return completion.choices[0].message.content

'''
 def get_embedding(text: str, model="text-embedding-3-small"):
    return openai.Embedding.create(input=text, model=model)['data'][0]['embedding']
'''

def get_tfidf(text, tfidf_model = TfidfVectorizer()):
    # 입력된 텍스트를 TF-IDF 특징 벡터로 변환
    text_vector = svm_model.predict([text])
    return text_vector

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

recreate_user_vector = False
if not os.path.isfile('user_vector.pkl'):
    recreate_user_vector = True

if recreate_user_vector:    
    user_vector = []
else:
    try:
        with open('user_vector.pkl', 'rb') as file:
            user_vector = pickle.load(file)
    except Exception as e:
        print("Error loading pickle file:", e)
        user_vector = []  # 파일을 읽을 수 없는 경우 빈 리스트로 초기화

#사용자 텍스트 입력
user_text = input()
'''
# MBTI 예측 모델을 통해 사용자의 MBTI 예측
print(user_text_translated)
user_text_vector = get_tfidf(user_text_translated)  # 벡터화

# 사용자 대화 벡터를 배열에 저장
user_vector.append(user_text_vector)

# 배열에 저장된 사용자 대화 임베딩을 NumPy 배열로 변환
user_embeddings = np.array(user_embeddings)
'''
# MBTI 예측 
# user_vector에 사용자 입력을 계속 추가하여 MBTI를 갱신
user_vector.append(translate_to_english(user_text))
combined_text = ' '.join(map(str, user_vector))  # 두 개의 텍스트를 공백을 이용하여 통합
combined_prediction = svm_model.predict([combined_text])  # 통합된 텍스트를 모델에 입력하여 예측 #정확도가 가장 높은 mbti를 결과로 반환

# all_messages 생성
recreate_all_messages=False
if not os.path.isfile('all_messages.pkl'):
    recreate_all_messages=True
    
if recreate_all_messages:    
    all_messages = [
    {"role": "system", "content": 
     f'''
     사용자의 MBTI 유형은 {combined_prediction}입니다. 
     당신은 상담모델로 사용자의 MBTI를 고려하여 답변하시길 바랍니다. 
     사용자는 한국인으로 한글로 답변해주시길 바랍니다.
     '''},
    {"role": "user", "content": user_text}]
else:
    try:
        with open('all_messages.pkl', 'rb') as file:
            all_messages = pickle.load(file)
        all_messages.append({"role": "user", "content": user_text})
    except Exception as e:
        print("Error loading pickle file:", e)
        all_messages = []  # 파일을 읽을 수 없는 경우 빈 리스트로 초기화

completion  = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=all_messages
)

# 생성된 응답 출력
response= completion.choices[0].message.content
print("ChatGPT Response:")
print(response)

# Assistant 대화를 user_messages에 추가
all_messages.append({"role": "assistant", "content": response})
print(all_messages)
# 코드가 반복 실행되면서 all_messages, user_embeddings에 각각 텍스트들이 계속 저장
with open('all_messages.pkl', 'wb') as file:
    pickle.dump(all_messages, file)
with open('user_vector.pkl', 'wb') as file:
    pickle.dump(user_vector, file)