from nltk.sentiment import SentimentIntensityAnalyzer
# import nltk

# NLTK 라이브러리 설치 및 VADER lexicon 다운로드
# nltk.download('vader_lexicon')

# 감정 분석을 위한 함수
def analyze_sentiment(responses):
    sia = SentimentIntensityAnalyzer()
    sentiments = [sia.polarity_scores(response) for response in responses]
    return sentiments

# 가중 평균을 계산하는 함수
def calculate_weighted_average(scores, weights):
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

# 상황에 대한 적절한 대답 평가
situation_score = 0.76 

# 텍스트 평가 점수 (GPT-4o)
text_eval_score = 0.7983  # 평균 점수

# MBTI 예측 모델의 정확도
mbti_accuracy = 0.83

# 피드백 감정 분석 점수 계산(한글=>영어)
bot_responses = [
    "The counseling experience was very meaningful as the counseling content matched well with the MBTI prediction model.",
    "The solution proposed based on MBTI was right for me, and it helped me a lot after the consultation.",
    "Counseling based on MBTI predictions was very helpful in deeply understanding my problems and finding solutions.",
    "The MBTI results accurately reflected my personal situation and the counseling offered was very helpful.",
    "Counseling based on MBTI tendencies was a great help in improving my self-psychological understanding.",
    "Counseling based on MBTI predictions was a great help in deeply understanding my problems and finding solutions.",
    "If MBTI is displayed as a classification, it may have a psychological impact on users. I am curious as to whether this will have an effect on accurate MBTI inference.",
    "It was said that MBTI changes depending on the situation, but I am concerned that if the MBTI is inferred by accumulating everything, the MBTI inference may become inaccurate if the conversation becomes long.",
    "The advantage of operating a consultation service is that we produce our own moderation rather than using what is provided by GPT for verification of hazards.",
    "Features for each MBTI trait were added to the prompt. The GUI was well organized and looked good.",
    "It's a shame that the function ends by simply guessing the MBTI. I think the quality will be better if you can check additional information in addition to the MBTI using the conversation content.",
    "As this is a service that emphasizes privacy, it would be nice to have a specific privacy protection policy.",
    "It's interesting that MBTI is predicted in real time as the conversation progresses.",
    "The technical difference lies in constructing a separate model for predicting MBTI and resolving imbalances in the dataset to improve its completeness.",
    "I felt that it was good to visualize the MBTI model, but I think it would be good to display probabilities for other MBTIs rather than just one MBTI."
]

# 사용자의 피드백을 통해 감정 분석
bot_sentiments = analyze_sentiment(bot_responses)

# 감정 분석 점수 (compound 점수를 사용하여 0~1 범위로 정규화)
bot_compound_scores = [(s['compound'] + 1) / 2 for s in bot_sentiments]

# 감정 분석 점수의 평균
sentiment_score = sum(bot_compound_scores) / len(bot_compound_scores)

# 점수 
scores = [
    situation_score, 
    text_eval_score, 
    sentiment_score, 
    mbti_accuracy, 
]

# 가중치 
weights = [1, 2, 2, 2]

# 가중 평균 계산
final_score = calculate_weighted_average(scores, weights)
print(f"Convopersona 성능평가점수: {final_score:.4f}/1.0")
