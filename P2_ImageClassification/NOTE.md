# Notes:

Here, I mentioned all the problems I faced in this project, their cause and solution.


1.) DL Model not getting load properly.

## Issue: 

- My model is a Sequential wrapper around a pre-trained VGG16 (as Functional) + Flatten + Dense layers.

- This is totally fine but the error means that when loading, Keras is trying to pass a list instead of a tensor into the Flatten layer.


## Reason: Why This Happens ?

- When saving Sequential models that include another Functional model (like VGG16), sometimes the "inputs" aren’t preserved correctly, and Keras reload gives a list instead of a tensor.


## Solution:

- Wrap the DL model in Functional API before saving.

- Instead of saving directly as Sequential, rebuild into a proper Functional model (which avoids the list issue):

Example :

```
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))

inputs = base_model.input
x = Flatten()(base_model.output)
x = Dense(256, activation="relu")(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs, outputs)
```

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2.) DL model Weights not loading for the VGG16 model.

# Issue: 

- Problem in running `model_training.py` file as VGG-16 model weights are not getting loaded.


# Cause:

- That error means Keras tried to download the VGG16 weights from Google Storage during training and the HTTPS connection got reset by something on your machine/network (firewall/ISP/VPN/antivirus/proxy). 


# Solution:

#### Step1: Download once manually, then load from disk (most reliable).

- Download the file in your browser (avoid python’s downloader) using this link: 
`https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5`

- Filename Keras expects: vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5


#### Step2: Put the file where Keras will find it.

i.) `Keras cache folder (automatic)` like ```C:\Users\MONIKA\.keras\models\```

- If .keras\models\ doesn’t exist, create it manually.

- After this, you don’t need to change your code — Keras will load it when you call:
```
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
```

ii.) Keep it inside your project (manual load)

- Create a model folder in your project:
```
C:\Users\MONIKA\Deep_Learning_Projects\ComputerVision_Projects\P2_ImageClassification\model\
```

- Move the .h5 file there.

- Modify your training code to load explicitly:

```
import os
from tensorflow.keras.applications import VGG16

LOCAL_WEIGHTS = os.path.join("model", "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")

base_model = VGG16(weights=LOCAL_WEIGHTS, include_top=False, input_shape=(224,224,3))
```

- Run training again

- Now when you run:
```
python model_training.py

```
- It will load weights from disk → no internet needed → no [WinError 10054] errors.


Tip: If you put it under C:\Users\MONIKA\.keras\models\, you can just keep weights="imagenet" and Keras will reuse the cached file automatically. (That’s Keras’ default cache directory on Windows.)



- Here, I am using (Option ii.) the local weights file instead of downloading from the internet.

- This is to avoid issues with internet connectivity during training.