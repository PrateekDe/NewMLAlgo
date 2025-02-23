import pandas as pd 
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

sentences = [
    "I am going to the store",
    "I am going to school",
    "She is reading a book",
    "He is playing football",
]


n = 4
data = []
for sentence in sentences:
    words = sentence.split()
    # print(words)
    # print(len(words))
    for i in range(len(words) -n):
        data.append((" ".join(words[i:i+n-1]), words[i+n-1]))
print(data)
df = pd.DataFrame(data, columns=["Previous Words", "Next Word"])
print(df)


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Previous Words"])
y = df["Next Word"]

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


sample_text = "I am going to"
sample_vector = vectorizer.transform([sample_text])
predicted_word = model.predict(sample_vector)


print(f"Predicted Next Word: {sample_text} {predicted_word[0]}")