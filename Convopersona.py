from openai import OpenAI
client = OpenAI(api_key="OPENAI_API_KEY")

# 영어로 번역하는 함수
def translate_to_english(text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content":f"Translate the following Korean text to English: '{text}'"}],
        #temperature=0,
        #top_p=1,
        #max_tokens=1500
    )
    return response.choices[0].text.strip()
    
def get_embedding(text, model="text-embedding-3-small"):
   # text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

all_messages = [{"role": "system", "content": ""}]
user_text = input()

# 사용자 대화를 user_messages에 추가
all_messages.append({"role": "user", "content": user_text})

# MBTI 예측 모델을 통해 사용자의 MBTI 예측
user_text_translated = translate_to_english(user_text)  # 입력 텍스트를 영어로 번역
user_text_embed = get_embedding(user_text_translated)  # 임베딩

# 사용자 대화 임베딩을 배열에 저장
user_embeddings = []
user_embeddings.append(user_text_embed)

# 배열에 저장된 사용자 대화 임베딩을 NumPy 배열로 변환
user_embeddings = np.array(user_embeddings)

# MBTI 예측 
# user_embeddings에 사용자 입력을 계속 추가하여 MBTI를 갱신
predicted_mbti = svm_model.predict(user_embeddings) #정확도가 가장 높은 mbti를 결과로 반환

completion  = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=all_messages
)

# 생성된 응답 출력
print("ChatGPT Response:")
print(completion.choices[0].text.strip())

# Assistant 대화를 user_messages에 추가
all_messages.append({"role": "assistant", "content": completion.choices[0].text.strip()})
