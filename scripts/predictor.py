from cnn_model import build_model
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np


class Predictor:

    def __init__(self, classes):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.model = None
        self.classes = classes
        self.session = tf.Session(config=config)
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                print("NN initialized")

    def load(self, model_path=None, target=448):
        with self.graph.as_default():
            with self.session.as_default():
                model = build_model(target, target, classes=len(self.classes))

                optimizer = Adam(learning_rate=0.01)
                model.compile(loss='categorical_crossentropy',
                              optimizer=optimizer,
                              metrics=['accuracy'])
                model.load_weights(model_path)
                self.model = model
                print("NN model loaded")

    def prepare_image(self, image, target):
        # Convert image to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # resize and preprocess image
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def decode_predictions(self, preds):
        predictions = map(str, (preds[0]*100).round(decimals=4))
        dictionary = dict(zip(self.classes, predictions))
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:5]
        return dict(sorted_dict)

    def predict(self, image, target):
        image = self.prepare_image(image, target)

        with self.graph.as_default():
            with self.session.as_default():
                preds = self.model.predict(image)
        results = self.decode_predictions(preds)

        return results

