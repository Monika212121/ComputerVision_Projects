import os, sys

from logger import logging
from exception import CustomException

from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Flatten, Dense 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
# ---- Handle TensorFlow oneDNN warning ----
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---- Suppress protobuf mismatch warnings ----
warnings.filterwarnings("ignore", message="Protobuf gencode version")

class TrainingPipeline:
    def __init__(self):
        self.train_dataset_path = 'dataset/train'
        self.valid_dataset_path = 'dataset/valid'

    def train(self):
        try:
            logging.info("train(): STARTS")
            
            # Load base VGG16 model (without top layers / last 3 dense layers)
            # base_model = VGG16(input_shape = (224, 224, 3), weights = 'imagenet', include_top = False)

            # NOTE: Download the weights once manually, then load from disk. Check in NOTE.md file (Issue2)
            LOCAL_WEIGHTS = os.path.join("model_weights", "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
            base_model = VGG16(weights=LOCAL_WEIGHTS, include_top=False, input_shape=(224,224,3))

            # Freeze convolutional layers / base model layers for transfer learning
            base_model.trainable = False 

            # NOTE: Using Functional API to create model instead of Sequential API. Check in NOTE.md file (Issue1)
            # Add custom classification layers
            x = Flatten()(base_model.output)
            x = Dense(256, activation = 'relu')(x)
            outputs = Dense(2, activation = 'softmax')(x)

            # Create the modified model / Functional API model
            model = Model(inputs = base_model.input, outputs = outputs)

            logging.info(f"Modified model summary: {model.summary()}")

            # Compiling
            model.compile(optimizer = Adam(learning_rate = 1e-4),
                        loss = 'categorical_crossentropy',
                        metrics = ['accuracy'])

            logging.info("Model compiled successfully.")

            # Preprocessing the training and validation images
            train_generator = ImageDataGenerator(rescale = 1./ 255,
                                                shear_range = 0.2,
                                                zoom_range = 0.2,
                                                horizontal_flip = True)

            validation_generator = ImageDataGenerator(rescale = 1./ 255)


            train_set = train_generator.flow_from_directory(self.train_dataset_path,
                                                            target_size = (224, 224),
                                                            batch_size = 32,
                                                            class_mode = 'categorical')

            valid_set = validation_generator.flow_from_directory(self.valid_dataset_path,
                                                                target_size = (224, 224),
                                                                batch_size = 32,
                                                                class_mode = 'categorical')

            logging.info("Training and validation data preprocessed successfully.")
            logging.info(f"Training set length: {len(train_set)}, Validation set length: {len(valid_set)}")

            # Training the model
            history = model.fit(train_set, 
                                validation_data = valid_set,
                                steps_per_epoch = len(train_set),
                                validation_steps = len(valid_set),
                                epochs = 10)
            logging.info("Model trained successfully.")

            # Create a folder to save the model if it doesn't exist
            if not os.path.exists('dl_model'):
                os.makedirs('dl_model')

            # Save the modified model
            model.save('dl_model/vgg16_mod_model.keras')
            logging.info("Model saved successfully.")
                        
            logging.info("train(): ENDS")


        except CustomException as e:
            logging.info(f"Error occurred in train(): {e}")
            raise CustomException(e, sys)