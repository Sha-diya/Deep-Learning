import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

df =pd.read_csv('Placement.csv')

print(df.shape)
print(df.head())


sns.scatterplot(x='cgpa', y='iq', hue='placement', data=df)
plt.show()


X = df[['cgpa', 'iq']].values  # Feature matrix (2D)
y = df['placement'].values      # Target array (1D)
scaler = StandardScaler()
X = scaler.fit_transform(X)

p=Perceptron()

p.fit(X,y)
print(p.coef_)#values of weights(w)
print(p.intercept_)#value of intercept(b)

plot_decision_regions(X, y, clf=p, legend=2)