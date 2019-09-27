from cnn_model import build_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.optimizers import Adam
import numpy as np


class Predictor:

    def __init__(self, model_path):

        model = build_model(448, 448, classes=15)
        optimizer = Adam(learning_rate=0.01)

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.load_weights(model_path)
        self.model = model

    def prepare_image(self, image, target):
        # Convert image to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # resize and preprocess image
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        return image

    def predict(self, image, target):
        image = self.prepare_image(image, target)

        preds = self.model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data = {"success": False}
        data["predictions"] = []

        for imagenetID, label, prob in results[0]:
            r = {"label": label, "probability": float(prob)}
            data["predictions"].append(r)

        data["success"] = True

        return data

