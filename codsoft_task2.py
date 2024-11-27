#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

movie_data = pd.read_csv('Movies.csv', encoding='latin1')
movie_data = movie_data.dropna(subset=['Rating'])
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
target = 'Rating'
X = movie_data[features]
y = movie_data[target]

transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[('cat', transformer, features)])
model = RandomForestRegressor(n_estimators=8, random_state=0)


# In[8]:


pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


# In[12]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)

mean_y_test = np.mean(y_test)
maep = (mae / mean_y_test) * 100  

print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("Mean Absolute Error Percentage (MAEP): {:.2f}%".format(maep))


# In[9]:


def predict_movie_rating():
    print("Enter the details of the movie:")
    genre = input("Genre: ")
    director = input("Director: ")
    actor_1 = input("Actor 1: ")
    actor_2 = input("Actor 2: ")
    actor_3 = input("Actor 3: ")
    input_data = pd.DataFrame({
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor_1],
        'Actor 2': [actor_2],
        'Actor 3': [actor_3]
    })
    predicted_rating = pipeline.predict(input_data)
    print(f"Predicted Rating: {predicted_rating[0]:.2f}")

predict_movie_rating()


# In[ ]:




