from keras.models import model_from_json
import numpy as np

dataset = np.loadtxt('predict.csv', delimiter=",")

X = dataset[:, 0:19]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")

loaded_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# scores = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))

scores = loaded_model.predict_classes(X, verbose=0)
print(scores)
