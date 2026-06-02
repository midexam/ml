import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.DataFrame({
    "Name": ["Ajay", "Mark", "Sara", "Faiza", "Sachin",
             "Rahul", "Pooja", "Smith", "Lakshmi", "Michael"],
    "Age": [32, 40, 16, 34, 34, 40, 20, 15, 55, 15],
    "Gender": ["M", "M", "F", "F", "M",
               "M", "F", "M", "F", "M"],
    "Sport": ["Football", "Neither", "Cricket", "Cricket", "Neither",
              "Cricket", "Neither", "Football", "Football", "Football"]
})

gender_encoder = LabelEncoder()
data["Gender_encoded"] = gender_encoder.fit_transform(data["Gender"])
sport_encoder = LabelEncoder()
data["Sport_encoded"] = sport_encoder.fit_transform(data["Sport"])

X = data[["Age", "Gender_encoded"]].values
y = data["Sport_encoded"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_scaled, y)  
test_age = 5
test_gender = "F"
test_gender_encoded = gender_encoder.transform([test_gender])

X_test = np.array([[test_age, test_gender_encoded[0]]])
X_test_scaled = scaler.transform(X_test)

prediction = model.predict(X_test_scaled)
predicted_sport = sport_encoder.inverse_transform(prediction)
print("Predicted Sport for Angelina:", predicted_sport[0])

plt.figure(figsize=(8,6))
for sport in np.unique(y):
    mask = (y == sport)
    plt.scatter(X[mask, 0], X[mask, 1],
                label=sport_encoder.inverse_transform([sport])[0],
                s=100)
    
plt.scatter(test_age, test_gender_encoded[0],
            marker='*', color='black', s=200,
            label=f"Angelina (Predicted: {predicted_sport[0]})")

plt.xlabel("Age")
plt.ylabel("Gender (0=F, 1=M)")
plt.title("KNN Classification of Sports by Age and Gender")
plt.legend()
plt.show()