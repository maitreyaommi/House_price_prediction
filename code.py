import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
df = pd.read_csv("Housing.csv")

df.head()
df.info()
df.dropna(inplace=True)
sns.histplot(df["price"])
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()
X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("R2 Score:", r2_score(y_test, predictions))
pickle.dump(model, open("model.pkl", "wb"))