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

# 주관적 평가 점수 수집
moderation_score = 0.8  # 모더레이션 대처 
situation_score = 0.9  # 여러가지 상황별 대답 

# 텍스트 평가 점수 (GPT-4o)
text_eval_score = 0.7983  # 평균 점수

# MBTI 예측 모델의 정확도
mbti_accuracy = 0.83

# 감정 분석 점수 계산
bot_responses = [
    "그래 맞어",

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
print(f"Convopersona 지표평가 최종 점수: {final_score:.4f}/1.0")
