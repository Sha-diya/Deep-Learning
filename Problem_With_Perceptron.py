import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron

or_data=pd.DataFrame()
and_data=pd.DataFrame()
xor_data=pd.DataFrame()


or_data['input1']=[1,1,0,0]
or_data['input2']=[1,0,1,0]
or_data['output']=[1,1,1,0]

and_data['input1']=[1,1,0,0]
and_data['input2']=[1,0,1,0]
and_data['output']=[1,0,0,0]
   

xor_data['input1']=[1,1,0,0]
xor_data['input2']=[1,0,1,0]
xor_data['output']=[0,1,1,0]
   
#print(and_data)
#sns.scatterplot(x=and_data['input1'],y=and_data['input2'], hue=and_data['output'],s=200)
#print(or_data)
#sns.scatterplot(x=or_data['input1'],y=or_data['input2'], hue=or_data['output'],s=200)
#print(xor_data)
#sns.scatterplot(x=xor_data['input1'],y=xor_data['input2'], hue=xor_data['output'],s=200)

clf1=Perceptron()
clf2=Perceptron()
clf3=Perceptron()

clf1.fit(and_data.iloc[:,0:2].values, and_data.iloc[:,-1].values)
clf2.fit(or_data.iloc[:,0:2].values, or_data.iloc[:,-1].values)
clf3.fit(xor_data.iloc[:,0:2].values, xor_data.iloc[:,-1].values)

plt.figure(figsize=(15, 5))
print(clf1.coef_)
print(clf1.intercept_)
x=np.linspace(-1,1,5)
y=-x+1
plt.subplot(1, 3, 1)
plt.plot(x,y)
sns.scatterplot(x=and_data['input1'],y=and_data['input2'], hue=and_data['output'],s=200)
plt.title("And Dataset")


print(clf2.coef_)
print(clf2.intercept_)
x=np.linspace(-1,1,5)
y=-x+0.5
plt.subplot(1, 3, 2)
plt.plot(x,y)
sns.scatterplot(x=or_data['input1'],y=or_data['input2'], hue=or_data['output'],s=200)
plt.title("Or Dataset")

print(clf3.coef_)
print(clf3.intercept_)
x=np.linspace(-1,1,5)
y=-x+0.5
plt.subplot(1, 3, 3)
plt.plot(x,y)
sns.scatterplot(x=xor_data['input1'],y=xor_data['input2'], hue=xor_data['output'],s=200)
plt.title("Xor Dataset")

