# python scripts/server.py
import io
from predictor import Predictor
from flask import Flask, render_template, Response, request, jsonify
from flask_restful import Resource, Api
from PIL import Image

app = Flask(__name__)
api = Api(app)

model = None


def load_model():
    # load the pre-trained Keras model
    global model
    # currently just mocks a response
    # model = {
    #     'Pneumonia': 0.929,
    #     'No findings': 0.542,
    #     'Consolidation': 0.497,
    #     'Atelectasis': 0.234,
    #     'Infiltration': 0.129}
    model = Predictor(model_path="model.h5")


# noinspection PyUnresolvedReferences
class LandingPage(Resource):
    def get(self):
        return Response(render_template('index.html'), 200, mimetype='text/html')


class ProcessFiles(Resource):
    def post(self):
        # TODO: Do some processing
        # time.sleep(2)

        if request.files:
            print("Image received")
            image = request.files["image"]
            print(image)
            image = image.read()
            image = Image.open(io.BytesIO(image))
            print(image)
            result = Predictor.predict(image, (448, 448))

            return jsonify(result)


api.add_resource(LandingPage, '/')
api.add_resource(ProcessFiles, '/process')


if __name__ == "__main__":
    load_model()

    app.run(debug=True)
