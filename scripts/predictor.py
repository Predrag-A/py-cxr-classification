from keras.models import load_model
from keras.preprocessing import image
from cnn_model import build_model
import numpy as np
import json


class Predictor:

    def __init__(self):

        temp_model = build_model(488, 488, 15)
        temp_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        temp_model.load_weights('model.h5')
        self.model = temp_model

    def predict(self, image_path):
        img = image.load_img(image_path, target_size=(448, 448))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.

        return self.model.predict(img_tensor)


if __name__ == '__main__':

    img_path = 'train/Atelectasis/00000011_006.png'
    predictor = Predictor()
    predict = predictor.predict(img_path)
    print(predict)
