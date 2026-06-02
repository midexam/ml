import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = {
    'Outlook': ['sunny','sunny','overcast','rainy','rainy','rainy',
                'overcast','sunny','sunny','rainy','sunny','overcast',
                'overcast','rainy'],
    'Temp': ['hot','hot','hot','mild','cool','cool',
             'cool','mild','cool','mild','mild','mild',
             'hot','mild'],
    'Humidity': ['high','high','high','high','normal','normal',
                 'normal','high','normal','normal','normal',
                 'high','normal','high'],
    'Wind': ['weak','strong','weak','weak','weak','strong',
             'strong','weak','weak','weak','strong',
             'strong','weak','strong'],
    'Play Tennis': ['no','no','yes','yes','yes','no',
                    'yes','no','yes','yes','yes',
                    'yes','yes','no']
}
df = pd.DataFrame(data)

le = LabelEncoder()
for column in df.columns:
    df[column] = le.fit_transform(df[column])

x = df.drop('Play Tennis', axis=1)
y = df['Play Tennis']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10)

rf = RandomForestClassifier(n_estimators=10, criterion="gini", random_state=100)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
c_mat = confusion_matrix(y_test, y_pred)
c_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(c_mat)
print("\nClassification Report:")
print(c_rep)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=rf.feature_importances_, 
    y=x.columns, 
    hue=x.columns,     
    palette="viridis", 
    legend=False      
)
plt.title("Feature Importance")
plt.show()
plt.figure(figsize=(19,7))
plot_tree(
    rf.estimators_[0], 
    filled=True, 
    feature_names=list(x.columns), 
    class_names=["no", "yes"],
    rounded=True
)
plt.title("Visualizing Tree 0 from Random Forest")
plt.show()
