from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#dataset
df = pd.read_csv(r'Churn_Modelling.csv')

print(df.head())
df.info()
print(f"Duplicated rows: {df.duplicated().sum()}")
print(df['Exited'].value_counts())
print(df['Geography'].value_counts())
print(df['Gender'].value_counts())

#Remove Unnecessary columns
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

#encoding for categorical features
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
print(df.head())

#Split the dataset into features(X) and target(y)
X = df.drop(columns=['Exited'])
y = df['Exited']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"Training data shape: {X_train.shape}")

#Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)

#Build the neural network model
model = Sequential()
model.add(Dense(11, activation='relu', input_dim=11))#Hidden layer #1
model.add(Dense(11, activation='relu'))#Hidden layer #2
model.add(Dense(1, activation='sigmoid'))#Output layer

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

print(f"Summary: {model.summary()}")

#Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=1)

#Make predictions on the test set
y_log = model.predict(X_test_scaled)
y_pred = np.where(y_log > 0.5, 1, 0).flatten()

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
