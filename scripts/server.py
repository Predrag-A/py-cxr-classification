# python scripts/server.py
import io
from create_data import get_class_labels
from predictor import Predictor
from flask import Flask, render_template, Response, request, jsonify
from flask_restful import Resource, Api
from PIL import Image

app = Flask(__name__)
api = Api(app)

predictor = None


# noinspection PyUnresolvedReferences
class LandingPage(Resource):
    def get(self):
        return Response(render_template('index.html'), 200, mimetype='text/html')


class ProcessFiles(Resource):
    def post(self):
        if request.files:
            image = request.files["image"]
            image = image.read()
            image = Image.open(io.BytesIO(image))
            result = predictor.predict(image=image, target_size=(448, 448))
            return jsonify(result)


def load_predictor():
    global predictor
    classes = get_class_labels('cxr-data/ClassLabels.txt')
    predictor = Predictor(classes=classes)
    predictor.load(model_path="model.h5", weights_only=True)


api.add_resource(LandingPage, '/')
api.add_resource(ProcessFiles, '/process')


def run_server():
    load_predictor()

    app.run(debug=True)


if __name__ == "__main__":
    run_server()
