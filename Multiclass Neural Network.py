"""
Author: Rambod Azimi
This Python code will use a neural network to recognize the handwritten digits 0-9.
"""

# Importing the required libraries
import numpy as np
import tensorflow as tf
from autils import load_data

# Loading the dataset
X_train, Y_train = load_data()
"""
The dataset contains 5000 training examples of handwritten digits.
Each training example is a 20x20 pixel grayscale image of the digit.
Each pixel is represented by a floating-point number indicating the grayscale intensity at that location.
"""

# Model representation
"""
We are going to use 2 dense hidden layers with ReLU activations,
as well as an output layer with a linear activation. 
"""
tf.random.set_seed(1234) # for consistent results
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(400,)),
    tf.keras.layers.Dense(units=25, activation='relu', name='layer1'),
    tf.keras.layers.Dense(units=15, activation='relu', name='layer2'),
    tf.keras.layers.Dense(units=10, activation='linear', name='layer3')
])

# summarizing the model
model.summary()

# compiling the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# training the model
model.fit(X_train, Y_train, epochs=70)

# prediction
image_of_2 = X_train[1015] # a sample image of 2
prediction = model.predict(image_of_2.reshape(1,400))

prediction = tf.nn.softmax(prediction) # use softmax to convert the prediction into a probability

output = np.argmax(prediction)
print(f"Prediction digit: {output}")