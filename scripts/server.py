# python scripts/server.py
import io
import numpy as np
from create_data import get_class_labels
from predictor import Predictor
from flask import Flask, render_template, Response, request, jsonify
from flask_restful import Resource, Api
from PIL import Image

app = Flask(__name__)
api = Api(app)

# Configuration parameters
predictor = None
max_age = 94.0
multi_input_flag = True
weights_only_flag = False


# noinspection PyUnresolvedReferences
class LandingPage(Resource):
    def get(self):
        return Response(render_template('index.html'), 200, mimetype='text/html')


class ProcessFiles(Resource):
    def post(self):
        if request.files:
            image, vec = process_inputs(request, multi_input=multi_input_flag)
            result = predictor.predict(image=image, target_size=(448, 448), multi_input=multi_input_flag, vec=vec)
            return jsonify(result)


def load_predictor():
    global predictor
    classes = get_class_labels('cxr-data/ClassLabels.txt')
    predictor = Predictor(classes=classes)
    predictor.load(model_path="model-1.h5", weights_only=weights_only_flag)


def process_inputs(request_data, multi_input=False):
    image = request_data.files["image"]
    image = image.read()
    image = Image.open(io.BytesIO(image))
    vec = None
    if multi_input:
        age = float(request.form['age'])
        age = age/max(age, max_age)
        vec = np.asarray([[age, float(request.form['gender']), float(request.form['position'])]])
        return image, vec
    return image, vec


api.add_resource(LandingPage, '/')
api.add_resource(ProcessFiles, '/process')


def run_server():
    load_predictor()

    app.run(debug=True)


if __name__ == "__main__":
    run_server()
