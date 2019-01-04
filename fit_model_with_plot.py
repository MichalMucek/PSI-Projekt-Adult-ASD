from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from PlotLearning import PlotLearning

np.random.seed(1)

dataset = np.loadtxt('csv_result-Autism-Adult-Data-MODIFIED.csv', delimiter=",")

Y = dataset[:, 19]
X = dataset[:, 0:19]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=5)

plot = PlotLearning()

model = Sequential()
model.add(Dense(6, input_dim=19, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='softmax'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=10, epochs=2500, callbacks=[plot])
model.fit(x_train, y_train, batch_size=15, epochs=2500, validation_data=(x_test, y_test), callbacks=[plot])

scores = model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
