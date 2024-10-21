import tensorflow
from tensorflow import keras 
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense,Flatten

(X_train,y_train),(X_test,y_test)= keras.datasets.mnist.load_data()

print(X_train.shape)
print(X_test.shape)
print(y_train)

#plt.imshow(X_train[2])

X_train=X_train/255;
X_test=X_test/255

model = Sequential()

#2D-->1D
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))#softmax for multiclassification
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
history = model.fit(X_train,y_train,epochs=10,validation_split=0.2)
y_prob = model.predict(X_test)
y_pred=y_prob.argmax(axis=1)

print(accuracy_score(y_test,y_pred))

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
plt.subplot(1,3,3)
plt.imshow(X_test[0])
print(model.predict(X_test[0].reshape(1,28,28)).argmax(axis=1))