from resnet_model import build_resnet
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
import numpy as np


class Predictor:
    """
    Class containing a ResNet CNN model created using Keras
    """

    def __init__(self, classes):
        """
        Constructor for the Predictor class. Graph and session are used to avoid conflicts
        when using multi-threading on a Flask or Django server
        :param classes: List containing all class labels for predictions
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.model = None
        self.classes = classes
        self.session = tf.Session(config=config)
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            with self.session.as_default():
                print("NN initialized")

    def load(self, model_path=None, target_size=448, weights_only=False):
        """
        Method used to lead the Keras CNN model
        :param model_path: Path to the model or to the model weights h5 file
        :param target_size: Size of the images, needed if model is initialized using weights
        :param weights_only: Flag determining whether the model is initialized using an entire model or only weights
        """
        with self.graph.as_default():
            with self.session.as_default():
                if weights_only:
                    # Load model if only weights are available
                    model = build_resnet(target_size, target_size, classes=len(self.classes))

                    optimizer = Adam(learning_rate=0.01)
                    model.compile(loss='categorical_crossentropy',
                                  optimizer=optimizer,
                                  metrics=['accuracy'])
                    model.load_weights(model_path)
                else:
                    # Load model if entire model is available
                    model = load_model(model_path)
                self.model = model

    def prepare_image(self, image, target_size):
        """
        Method used to prepare images for processing
        :param image: Image opened using io.BytesIO
        :param target_size: Target size of the image
        :return: Processed image
        """
        # Convert image to grayscale if needed
        if image.mode != "L":
            image = image.convert("L")

        # Resize and preprocess image
        image = image.resize(target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return image

    def decode_predictions(self, preds):
        """
        Method used to take the prediction data from the model,
        pair it with corresponding labels and return top 5 results
        :param preds: Prediction data generated using model.predict()
        :return: Dictionary containing top five results with corresponding labels
        """
        predictions = map(str, (preds[0]*100).round(decimals=4))
        dictionary = dict(zip(self.classes, predictions))
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:5]
        return dict(sorted_dict)

    def predict(self, image, target_size):
        """
        Method used to return top 5 most likely classes of the image using a trained Keras model
        :param image: Image opened using io.BytesIO
        :param target_size: Target size of the image
        :return: Dictionary containing top five results with corresponding labels
        """
        image = self.prepare_image(image, target_size)

        with self.graph.as_default():
            with self.session.as_default():
                preds = self.model.predict(image)
        results = self.decode_predictions(preds)

        return results

