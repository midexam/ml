import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10)

rf = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=100)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
c_mat = confusion_matrix(y_test, y_pred)
c_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", c_mat)
print("\nClassification Report:\n", c_rep)

plt.figure(figsize=(8,5))
sns.barplot(
    x=rf.feature_importances_, 
    y=x.columns, 
    hue=x.columns,   
    palette="viridis", 
    legend=False      
)
plt.title("Feature Importance (Iris Dataset)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

plt.figure(figsize=(19,7))
plot_tree(
    rf.estimators_[0],
    filled=True,
    feature_names=list(x.columns),
    class_names=list(iris.target_names),
    rounded=True
)
plt.title("Visualization of Tree 0 in the Random Forest")
plt.show()