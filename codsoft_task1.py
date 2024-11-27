#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('Titanic-Dataset.csv')
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])




# In[14]:


imputer = SimpleImputer(strategy="mean")
data["Age"] = imputer.fit_transform(data[["Age"]])
data["Fare"] = imputer.fit_transform(data[["Fare"]])
data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

# Convert Embarked into one-hot encoding
data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)
X = data.drop(columns=["Survived"])
y = data["Survived"]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# In[16]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[17]:


def predict():
    print("Enter the following details to predict survival on the Titanic:")
    try:
        cls = int(input("Passenger Class (1, 2, or 3): "))
        gender = int(input("Sex (1 for male, 0 for female): "))
        age = float(input("Age: "))
        sib = int(input("Number of Siblings/Spouses Aboard: "))
        par = int(input("Number of Parents/Children Aboard: "))
        fare = float(input("Fare: "))
        emb_C = int(input("Embarked at Cherbourg (1 for Yes, 0 for No): "))
        emb_Q = int(input("Embarked at Queenstown (1 for Yes, 0 for No): "))

        features = [[cls, gender, age, sib, par, fare, emb_C, emb_Q]]

        result = model.predict(features)
        if result[0] == 1:
            print("Prediction: The passenger would have survived.")
        else:
            print("Prediction: The passenger would not have survived.")
    except Exception as e:
        print("Error in input. Please try again.", str(e))

predict()


# In[ ]:




