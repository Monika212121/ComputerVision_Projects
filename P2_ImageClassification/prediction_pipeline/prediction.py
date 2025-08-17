import os, sys
import numpy as np

import logging
from exception import CustomException


import warnings
# ---- Handle TensorFlow oneDNN warning ----
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---- Suppress protobuf mismatch warnings ----
warnings.filterwarnings("ignore", message="Protobuf gencode version")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



class PredictPipeline:
    def __init__(self, input_image_path):
        self.input_image_path = input_image_path


    def predict(self):
        try:
            
            logging.info("predict(): STARTS")

            # Loading the model
            model = load_model(os.path.join('dl_model', 'vgg16_mod_model.keras'), compile = False)

            # Preprocssing the input image
            img_file_path = self.input_image_path
            logging.info(f"Input image file path: {img_file_path}")

            input_img = image.load_img(img_file_path, target_size = (224, 224))
            input_img = image.img_to_array(input_img)                                       # Due to this step encodeImage not needed
            input_img = np.expand_dims(input_img, axis = 0)

            # Predicting the class of the input image
            prediction = model.predict(input_img)                                           # Here, this predit() is model's method

            logging.info(f"Prediction: {prediction}")
            
            # Getting the predicted class [cat, dog]
            result = np.argmax(prediction, axis = 1)
            logging.info(f"Prediction result: {result}")

            # Returning the predicted class in the UI

            # case1: Neither cat or dog                                                       
            threshold = 0.5                                                                 # Threshold for classification confidence                                                             
            if prediction[0][0] < 0.5 and prediction[0][1] < 0.5:                           # prediction like [[0.2, 0.8]]
                category = 'Other than Cat or dog'
                logging.info(f"Prediction result category: {category}")
                logging.info("predict(): ENDS")
                return [{'Classification': category}]

            # case2: Cat
            if result[0] == 0:
                category = 'Cat'
                logging.info(f"Prediction result category: {category}")
                logging.info("predict(): ENDS")
                return [{'Classification': category}]
            
            # case3: Dog
            else:
                category = 'Dog'
                logging.info(f"Prediction result category: {category}")
                logging.info("predict(): ENDS")
                return [{'Classification': category}]


        except Exception as e:
            logging.info(f"Error occurred in predict(): {e}")
            raise CustomException(e, sys)
