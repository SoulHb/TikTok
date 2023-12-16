import cv2
import io
import numpy as np
from flask import Flask, request, jsonify
import torch
import argparse
import albumentations as A
import sys
from albumentations.pytorch import ToTensorV2
sys.path.append('src')
from model.model import Unet
from PIL import Image
from config.config import *
def load_model(model_path):
    model = Unet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Define parser
parser = argparse.ArgumentParser()
parser.add_argument("--saved_model_path", type=str,
                    help='Specify path for chosen model')
args = parser.parse_args()

# Define model
transform = {'image': A.Compose([A.Resize(736, 384), A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ToTensorV2(),]),
             'mask': A.Compose([A.Resize(736, 384), ToTensorV2(),])}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(args.saved_model_path if args.saved_model_path else f"{SAVED_MODEL_FOLDER}/model.pth")
app = Flask(__name__)


def add_mask(img, mask):
    """
        This function create yellow mask for processed image.
        Args:
            img (np.array): image
            mask (np.array): mask
        Returns:
            np.array: processed image with yellow mask
    """
    mask = mask.astype(np.uint8)

    # create yellow mask
    mask_yellow = np.concatenate([mask * 255, mask * 255, mask * 0], axis=2)

    # create mask of a person
    mask_person = cv2.bitwise_and(img, img, mask=mask)

    # create mask of background
    mask_background = cv2.bitwise_and(img, img, mask=cv2.threshold(cv2.bitwise_not(mask), 0, 255, cv2.THRESH_OTSU)[1])

    # add yellow color to a human mask
    dst = cv2.addWeighted(mask_person, 0.6, mask_yellow, 0.4, 0.0)

    # add human mask and background mask to ger resulted image
    return cv2.add(dst, mask_background)


@app.route('/process_image', methods=['POST'])
def process_image():
    """
        This function processes an uploaded image, makes a prediction using a model,
        and returns the result with a yellow mask.
        Args:
            None

        Returns:
            JSON: {'prediction': result} where result is a list representation of the processed image.
        """
    # Read the image file from the request
    file = request.files['file']
    image_bytes = file.read()

    # Open the image using PIL and convert it to a NumPy array
    image = Image.open(io.BytesIO(image_bytes))
    image = np.array(image)

    # Make a copy of the original image
    image_1 = image.copy()

    # Normalize the image values to the range [0, 1]
    image = image / 255

    # Apply a transformation to the image
    transformed_image = transform['image'](image=image)
    image = transformed_image['image'].to(device)

    # Get the shapes of the original and transformed result
    image_shape = image.shape
    image_1_shape = image_1.shape

    # Resize the original image to match the transformed image size
    image_1 = cv2.resize(image_1, (image_shape[2], image_shape[1]))

    # Make a prediction using the model
    predicted = model(image.unsqueeze(0))

    # Convert the prediction to binary values (0 or 1) using a threshold of 0.5
    predicted = predicted > 0.5

    # Rearrange dimensions and convert the prediction to a NumPy array
    predicted = torch.permute(predicted.squeeze(0), (1, 2, 0))
    predicted = predicted.cpu().detach().numpy().astype(np.uint8)

    # Add the predicted mask to the original image
    result = add_mask(image_1, predicted)

    # Resize the result to match the original image size
    result = cv2.resize(result, (image_1_shape[1], image_1_shape[0]))

    # Return the result as JSON
    return jsonify({'prediction': result.tolist()})


@app.route('/process_video', methods=['POST'])
def process_video():
    """
            This function processes an uploaded video, makes a prediction using a model,
            and returns the result video with a yellow mask.
            Args:
                None

            Returns:
                Bytes: processed video.
            """
    video = request.files['file']
    video.save(video.filename)
    cap = cv2.VideoCapture(video.filename)
    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc('V','P','8','0')
    out = cv2.VideoWriter('output.webm', fourcc, 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        # Read a frame from the input video
        ret, image = cap.read()
        if not ret:
            break
        image_1 = image.copy()
        image = image / 255
        transformed_image = transform['image'](image=image)
        image = transformed_image['image'].to(device)
        image_shape = image.shape
        image_1_shape = image_1.shape
        image_1 = cv2.resize(image_1, (image_shape[2], image_shape[1]))
        predicted = model(image.unsqueeze(0))
        predicted = predicted > 0.5
        predicted = torch.permute(predicted.squeeze(0), (1, 2, 0))
        predicted = predicted.cpu().detach().numpy().astype(np.uint8)
        result = add_mask(image_1, predicted)
        result = cv2.resize(result, (image_1_shape[1], image_1_shape[0]))

        out.write(result)

    # Release input and output video objects
    cap.release()
    out.release()
    with open('output.webm', 'rb') as f:
        result = f.read()
    return result


app.run(host='0.0.0.0', debug=False)

