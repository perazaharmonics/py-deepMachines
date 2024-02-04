import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Load the MobileNet model
model = tf.keras.applications.MobileNet(weights='imagenet')

# Create an ArgumentParser object
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument('--im_path', type=str, default='data/1.jpg', help='path to the image')
args = parser.parse_args()
# Get the path to the image
IM_PATH = args.im_path
# Read the image
img = image.load_img(IM_PATH, target_size=(224, 224))
# Convert the pixels to a numpy array
img = image.img_to_array(img)
# Expand the dimensions of the array
img = np.expand_dims(img, axis=0)
# Preprocess the input image
img = preprocess_input(img)
# Make predictions
predictions = model.predict(img)
# Decode the predictions
output = decode_predictions(predictions)
# Print the output
print(output)

# Display the image
plt.imshow(img[0])
plt.axis('off')
plt.show()
