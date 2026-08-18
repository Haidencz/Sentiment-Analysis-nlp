from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data_path = Path("data/aclImdb/train")
positive_path = data_path/"pos"
negative_path = data_path / "neg"

print("+ reviews:", len(list(positive_path.glob("*.txt")))) #number of pos reviews
print("- reviews:", len(list(negative_path.glob("*.txt")))) #number of neg reviews

positive_files = list(positive_path.glob("*.txt")) #
negative_files = list(negative_path.glob("*.txt")) 
with open(positive_files[0], "r", encoding="utf-8") as file: positive_review = file.read()
with open(negative_files[0], "r", encoding="utf-8") as file: negative_review = file.read()
print("\n random + review:")
print(positive_review[:500]) #show a pos review
print("\n random - review:")
print(negative_review[:500]) #show a neg review

reviews = []
labels = []
for file_path in positive_files:
    with open(file_path, "r", encoding="utf-8") as file:
        review_text = file.read()
    reviews.append(review_text)
    labels.append(1) #pos loop & store
for file_path in negative_files:
    with open(file_path, "r", encoding="utf-8") as file:
        review_text = file.read()
    reviews.append(review_text)
    labels.append(0)  #neg loop and store
df = pd.DataFrame({"review": reviews, "sentiment": labels})
print(df.head())
print(df.tail())
print(df["sentiment"].value_counts())
X = df["review"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2) #create vectorizer
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(X_train_tfidf.shape) #features
print(X_test_tfidf.shape) #features

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_test_tfidf)
print(predictions[:10]) #predict the first 10 reviews
print(y_test.iloc[:10].values) #the actual answerss (compare the 1s and 0s if same = good!!)
accuracy = accuracy_score(y_test, predictions)
print("accuracy:", accuracy)
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions) 
print(cm)
##ignore just evaluatory
wrong_predidctions = X_test[predictions != y_test]
print("\n Misclassifications:")
for index in wrong_predidctions.index[:5]:
    print("\n review:")
    print(X_test[index][:500])
    print("Actual:", y_test[index])
    print("predicted:", predictions[list(X_test.index).index(index)])
##ignore just evaluatory
print("training reviews:", len(X_train)) #20k
print("test reviews:", len(X_test)) #5000

print(len(reviews)) #should print 25000
print(len(labels)) #should print 25000
print("first label: ", labels[0])
print("first review: ", reviews[0][:100])
print("last label: ", labels[-1])
print("last review: ", reviews[-1][:100]) #san check