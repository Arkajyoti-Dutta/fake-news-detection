print("===== FAKE NEWS DETECTION SYSTEM =====")

# Importing libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading datasets
fake_data = pd.read_csv("Fake.csv")
true_data = pd.read_csv("True.csv")

# Adding labels
fake_data["label"] = 0
true_data["label"] = 1

# Combining datasets
data = pd.concat([fake_data, true_data])

# Shuffling dataset
data = data.sample(frac=1)

# Resetting index
data.reset_index(drop=True, inplace=True)

# Combining title and text
data["content"] = data["title"] + " " + data["text"]

# Keeping necessary columns
data = data[["content", "label"]]

# Input and output data
x = data["content"]
y = data["label"]

# Splitting training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25
)

# Converting text into numerical values
vectorizer = TfidfVectorizer(stop_words='english')

xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Creating and training model
model = LogisticRegression()

model.fit(xv_train, y_train)

# Checking accuracy
predictions = model.predict(xv_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Taking user input
news = input("\nEnter News: ")

# Transforming input news
news_vector = vectorizer.transform([news])

# Predicting result
prediction = model.predict(news_vector)

# Displaying output
if prediction[0] == 0:
    print("\n❌ This news appears to be FAKE.")
else:
    print("\n✅ This news appears to be REAL.")

print("\nThank you for using Fake News Detection System")

