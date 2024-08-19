from tensorflow.keras.datasets import mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.astype('float64')
test_x = test_x.astype('float64')

train_x /= 255
test_x /= 255

model = Sequential()

# Input layer
input_layer = Flatten(input_shape=(train_x[0].shape))
model.add(input_layer)

# Hidden layer
hidden_layer_1 = Dense(256, activation='sigmoid')
hidden_layer_2 = Dense(128, activation='sigmoid')

model.add(hidden_layer_1)
model.add(hidden_layer_2)

# Output layer
output_layer = Dense(10, activation='sigmoid')
model.add(output_layer)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100, batch_size=2000, validation_split=0.2)

model.save('digit_classifier.h5')
model = load_model('digit_classifier.h5')

img = load_img('three.png', target_size=(28, 28), color_mode='grayscale')
plt.imshow(img, cmap='gray')
img = img_to_array(img)/255
img = img.T
img.shape
prediction=model.predict(img)
np.argmax(prediction)
idx = 743
test_im = test_x[idx]
prediction=model.predict(np.array([test_im]))
np.argmax(prediction), test_y[idx]