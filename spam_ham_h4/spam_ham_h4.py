import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
from nltk.stem import PorterStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv("https://raw.githubusercontent.com/Sanjay-dev-ds/spam_ham_email_detector/master/spam.csv")
df = df.drop_duplicates(keep='first')

x = df['EmailText'].values
y = df['Label'].values

porter_stemmer = PorterStemmer()

def preprocessor(text):
    text = text.lower()
    text = re.sub("\\W", " ", text)
    text = re.sub("\\s+(in|the|all|for|and|on)\\s+", " _connector_ ", text)
    words = re.split("\\s+", text)
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]
    return ' '.join(stemmed_words)

def tokenizer(text):
    text=re.sub("(\\W)"," \\1 ",text)
    return re.split("\\s+",text)

vectorizer = CountVectorizer(tokenizer=tokenizer,ngram_range=(1,2),min_df=0.006,preprocessor=preprocessor)
x  = vectorizer.fit_transform(x)

ros = RandomOverSampler(random_state=42)
print('Original dataset shape', Counter(y))

x, y = ros.fit_resample(x, y)

print('Modified dataset shape', Counter(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = SVC(C=1, kernel="linear")
model.fit(x_train, y_train)

accuracy = metrics.accuracy_score(y_test, model.predict(x_test))
accuracy_percentage = 100 * accuracy
print(accuracy_percentage)

params  = {"C":[0.2, 0.5] , "kernel" : ['linear', 'sigmoid'] }
cval = KFold(n_splits=2)
model = SVC()
TunedModel = GridSearchCV(model, params, cv=cval)
TunedModel.fit(x_train, y_train)

print(classification_report(y_test, TunedModel.predict(x_test)))

# matrix heatmap
sns.heatmap(confusion_matrix(y_test, TunedModel.predict(x_test)), annot=True, fmt="g")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Distribution of labels (Spam vs Ham)
plt.figure(figsize=(6, 4))
sns.countplot(x=df['Label'])
plt.title('Distribution of Spam vs Ham Emails')
plt.xlabel('Label (Spam/Ham)')
plt.ylabel('Count')
plt.show()

mails = [
    "Hey, you have won a car !!!!. Conrgratzz",
    "Dear applicant, Your CV has been recieved. Best regards",
    "You have received $1000000 to your account",
    "Join with our whatsapp group",
    "Kindly check the previous email. Kind Regard"
]

for mail in mails:
    is_spam = TunedModel.predict(vectorizer.transform([mail]).toarray())
    print(mail + " : " + ("Spam" if is_spam[0] == 1 else "Ham"))
