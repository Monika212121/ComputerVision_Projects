from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from prediction_pipeline.prediction import PredictPipeline

from logger import logging
from utils import decode_image

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.imgFileName = 'inputImage.jpg'
        self.classifier = PredictPipeline(self.imgFileName)


@app.route('/', methods = ['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decode_image(image, clApp.imgFileName)
    result = clApp.classifier.predict()
    return jsonify(result)


if __name__ == '__main__':
    logging.info("Logging has started: main()")
    clApp = ClientApp()
    app.run(host = '0.0.0.0', port = 8080, debug = True)