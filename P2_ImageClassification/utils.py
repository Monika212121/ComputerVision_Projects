import base64
import logging

# Encodes an image to base64.
def encode_image(imagePath):
    with open(imagePath, 'rb') as f:
        logging.info(f"Encoding image: {imagePath}")
        return base64.b64encode(f.read())


# Decodes base64 to the image.
def decode_image(imgBase64String, outputImageFilePath):
    imgData = base64.b64decode(imgBase64String)

    with open(outputImageFilePath, 'wb') as f:
        f.write(imgData)
        logging.info(f"Decoded image and saved to: {outputImageFilePath}")
        f.close()
