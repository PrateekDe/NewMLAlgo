'''MIT License

Copyright (c) 2025 Prateek De

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
THIS IS A PROPERTY OF MINDSCRIBE TECH.. PLEASE DO NOT COPY OR DISTRIBUTE WITHOUT PERMISSION
'''



# Install all the necessary libraries using: pip install -r requirements.txt

import pandas as pd
import numpy as np
import sklearn as sklearn
from datasets import load_dataset
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV

# Breaking the existing dataset into 3 sectors: Train, Validation, Tests
data = load_dataset('go_emotions')
df_train = pd.DataFrame(data["train"])
df_val = pd.DataFrame(data["validation"])
df_test = pd.DataFrame(data["test"])


# Mostly for the data preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://S+|www\.S+', '', text) #This removes URLs
    text = re.sub(r'\S*@\S*\s?', '', text)  # Remove emails
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df_train["text"] = df_train["text"].apply(preprocess)
df_val["text"] = df_val["text"].apply(preprocess)
df_test["text"] = df_test["text"].apply(preprocess)


# Next, we wanna use vectorization to convert the text data into numerical data.. Since ML algos dont understand 
vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1,2))  # Bigram TF-IDF
X_train_tfidf = vectorizer.fit_transform(df_train["text"])
X_val_tfidf = vectorizer.transform(df_val["text"])
X_test_tfidf = vectorizer.transform(df_test["text"])


# Now we wanna convert the labels into a multi-label format
from sklearn.preprocessing import MultiLabelBinarizer

# Convert lists into multi-label format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(df_train["labels"])
y_val = mlb.transform(df_val["labels"])
y_test = mlb.transform(df_test["labels"])






param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [20, 30, 40],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="f1_weighted")
grid_search.fit(X_train_tfidf, y_train)

print("Best Params:", grid_search.best_params_)





# Next, we wanna split the data into training and Validation sets.. Now this is really important since we wanna make sure that everything is gonna work perfecty  
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)




# Now we wanna train the model.. We are gonna use the Random Forest Classifier
# Initialize Random Forest Model
rf_model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)



# Checking the model performance  
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")


