
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("data.csv")
X = df[['engine_size', 'distance']]
y = df["load"]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
print("Model trained successfully")
