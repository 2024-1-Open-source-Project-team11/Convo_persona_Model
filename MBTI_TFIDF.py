import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


# MBTI 데이터셋 불러오기
mbti_data = pd.read_csv("MBTI 500.csv")

# 벡터화
tfidf = TfidfVectorizer()

# 데이터셋을 학습용과 테스트용으로 나눔
X_train, X_test, y_train, y_test = train_test_split(mbti_data['posts'], mbti_data['type'], test_size=0.2, random_state=42)
X_train_tfidf = tfidf.fit_transform(X_train)

'''
# X_train_tfidf 행렬의 각 행마다 0이 아닌 원소의 개수를 세는 코드
non_zero_counts = X_train_tfidf.getnnz(axis=1)

# 각 행의 0이 아닌 원소의 개수를 출력
for i, count in enumerate(non_zero_counts):
    print(f"행 {i+1}의 0이 아닌 원소 개수: {count}")
'''

# 서포트 벡터 머신 분류 모델 학습
clf = LinearSVC(C=0.4)
clf.fit(X_train_tfidf, y_train)
cv = GridSearchCV(clf, {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1.0]}, scoring = "accuracy")

# 그리드 서치 모델 인스턴스화
text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',cv)])
text_clf.fit(X_train, y_train)

C = cv.best_estimator_.C

y_pred = text_clf.predict(X_test)

# 성능 평가
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)