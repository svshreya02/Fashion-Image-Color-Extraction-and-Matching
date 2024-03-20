from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch
import torchvision


app = FastAPI()

# Load the CSV file containing color data
color_df = pd.read_csv('/home/oem/Desktop/MSD/Task1/colours_rgb_shades_clean.csv')

# Load the trained model for prediction
model = torchvision.models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)  # Adjust the number of output classes as necessary
model.load_state_dict(torch.load('/home/oem/Desktop/MSD/Task1/torch.pth'))  # Replace 'model_path.pth' with the actual model path
model.eval()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification and Color Extraction API!"}

# Function to extract colors by removing the background
def extract_colors(image_path, num_colors=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binary thresholding
    _, binary_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours from the binary threshold image
    contours, hierarchy = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the largest contour
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, color=255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Extract the masked area for color analysis
    idx = np.where(mask != 0)
    pixels = image[idx[0], idx[1], :]

    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_

    # Count the occurrences of each label
    labels = kmeans.labels_
    counts = np.bincount(labels)

    return colors.astype(int), counts

# Function to convert RGB to HEX
def rgb_to_hex(rgb):
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'

def find_closest_hex_color(colors, color_df):
    closest_colors = []
    for color in colors:
        hex_color = rgb_to_hex(color)
        distances = color_df['RGB Hex'].apply(
            lambda hex_code: sum((int(hex_code[i:i+2], 16) - color[i//2])**2 for i in range(0, 6, 2))
            if hex_code and len(hex_code) == 6 else float('inf')
        )
        closest_index = distances.idxmin()
        if distances[closest_index] == float('inf'):
            closest_color = ('Invalid Color Code', '#')
        else:
            closest_color = color_df.iloc[closest_index]
        closest_colors.append((closest_color['Color Name'], closest_color['RGB Hex']))
    return closest_colors


@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image.save("temp_image.jpg")
    colors, counts = extract_colors("temp_image.jpg")

    closest_colors = find_closest_hex_color(colors, color_df)

    return JSONResponse(content={"closest_colors": closest_colors})

