import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--im_path', type=str, default='data/1.jpg', help='path to the image')
args = parser.parse_args()
IM_PATH = args.im_path

def read_image(imPath):
    img = cv2.imread(imPath)
    return img

# Path to the directory containing the saved model
MODEL_DIR = '/home/perazaharmonics/Python_ML/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'

# Load the TensorFlow SavedModel
loaded_model = tf.saved_model.load(MODEL_DIR)
infer = loaded_model.signatures['serving_default']

# now we will read the images in the specified directory path. Here, we are considering only
# .jpeg images, but you can change it into another format if you want:

for dirs in os.listdir(IM_PATH):
    if not dirs.startswith('.'):
        for im in os.listdir(os.path.join(IM_PATH, dirs)):
            if im.endswith('.jpeg'):
                image = read_image(os.path.join(IM_PATH, dirs, im))
                if image is None:
                    print('Image is read as None')
                else:
                    print('Image name:', im)

                # Preprocess the image as required by the model
                # This usually involves resizing the image and expanding its dimensions
                input_tensor = tf.convert_to_tensor(image)
                input_tensor = input_tensor[tf.newaxis,...]

                # Run inference
                detections = infer(input_tensor)

                # Extract boxes and scores
                boxes = detections['detection_boxes'][0].numpy()
                classes = detections['detection_classes'][0].numpy().astype(np.int32)
                scores = detections['detection_scores'][0].numpy()

                # Visualization of the results of a detection
                imageHeight, imageWidth, _ = image.shape
                for i, box in enumerate(boxes):
                    if scores[i] > 0.5:  # Confidence threshold
                        ymin, xmin, ymax, xmax = box
                        (left, right, top, bottom) = (xmin * imageWidth, xmax * imageWidth,
                                                      ymin * imageHeight, ymax * imageHeight)
                        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (255, 255, 0), thickness=1)
                        print('Class:', classes[i], 'Confidence:', scores[i])

                plt.figure(figsize=(10, 10))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.show()
