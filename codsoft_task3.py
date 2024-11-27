#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('IRIS.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]  


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)


# In[5]:


y_pred = forest_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[6]:



def predict_species():
    print("\nEnter flower measurements to predict the species:")
    try:
        sepal_length = float(input("Sepal Length (cm): "))
        sepal_width = float(input("Sepal Width (cm): "))
        petal_length = float(input("Petal Length (cm): "))
        petal_width = float(input("Petal Width (cm): "))
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        prediction = forest_clf.predict(input_data)
        print(f"The predicted species is: {prediction[0]}")
    except ValueError:
        print("Invalid input. Please enter numeric values for the measurements.")
predict_species()


# In[ ]:




