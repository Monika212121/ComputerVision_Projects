from training_pipeline.training import TrainingPipeline
import logging
import os

import warnings
# ---- Handle TensorFlow oneDNN warning ----
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# ---- Suppress protobuf mismatch warnings ----
warnings.filterwarnings("ignore", message="Protobuf gencode version")



if __name__ == '__main__':

    logging.info("main(): STARTS")

    model_trainer_obj = TrainingPipeline()
    model_trainer_obj.train()

    logging.info("main(): ENDS")