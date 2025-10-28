#import necessary libraries
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# load the data (MNIST image dataset)
(x_train,y_train),(x_test,y_test)=mnist.load_data() 
print(x_train.dtype)
print(x_train.shape)
print(y_test.shape) 
print(x_train[0]) 


#plt.imshow(x_train[0])
#plt.show()
#print("****************************")
#print(f"label is: {y_train[0]}")

#normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#to_categorical
print(f"before : label is: {y_train[100]}")
y_train = to_categorical(y_train)
print(f"after : label is: {y_train[100]}")

y_test = to_categorical(y_test)

#architect the model
model=Sequential()
model.add(Flatten(input_shape=(28,28))) #first layer
model.add(Dense(128,'relu')) #128 neurons
model.add(Dense(10,'softmax')) #10 neurons for 10 classes last layer

#compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) #accuracy on training dataset



#train
result=model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)#epochs means no of times entire data is passed v_split=0.2 take 20% data for validation or validation_data=(x_test,y_test)

#evaluate
loss,accuracy = model.evaluate(x_test, y_test)
print(f"test loss : {loss}")
print(f"test accuracy : {accuracy}")
print(result.history.keys())
print(result.history.values())
print(result.history)

#plotting visualizations
plt.plot(result.history['val_accuracy'], label="validate accuracy", color="blue")
plt.plot(result.history['accuracy'], label="train accuracy", color="green")
plt.title("train_accuracy vs val_accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()
