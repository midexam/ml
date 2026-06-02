import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

train_texts = [
    "Win money now",
    "Claim free prize",
    "Limited time offer",
    "Exclusive deal just for you",
    "Hello how are you",
    "Let's meet tomorrow",
    "Are you available for a call",
    "Good morning have a nice day"
]
train_labels = [1, 1, 1, 1, 0, 0, 0, 0] 
test_texts = [
    "Earn money fast from home",      
    "Urgent offer claim your prize",  
    "Hey friend let us meet up",     
    "Good morning call me later"      
]
test_labels = [1, 1, 0, 0] 

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_texts)
model = MultinomialNB()
model.fit(X_train, train_labels)
X_test = vectorizer.transform(test_texts)
y_pred = model.predict(X_test)
print("--- Test Set Predictions ---")
for text, true_lbl, pred_lbl in zip(test_texts, test_labels, y_pred):
    true_status = "Spam" if true_lbl == 1 else "Not Spam"
    pred_status = "Spam" if pred_lbl == 1 else "Not Spam"
    print(f"Text: '{text}'\n-> Actual: {true_status} | Predicted: {pred_status}\n")

cm = confusion_matrix(test_labels, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Spam", "Spam"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix (Test Data)")
plt.show()

test_spam_count = test_labels.count(1)
test_not_spam_count = test_labels.count(0)

plt.figure()
plt.bar(["Spam", "Not Spam"], [test_spam_count, test_not_spam_count], color=["red", "green"])
plt.title("Test Data: Spam vs Not Spam Count")
plt.ylabel("Number of Messages")
plt.show()