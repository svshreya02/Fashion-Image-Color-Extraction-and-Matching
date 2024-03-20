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

app = FastAPI()

# Load the CSV file containing color data
color_df = pd.read_csv('/home/oem/Desktop/MSD/Task1/colours_rgb_shades_clean.csv')

# Load the trained model for prediction
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 7)  # Adjust the number of output classes as necessary
model.load_state_dict(torch.load('/home/oem/Desktop/MSD/Task1/torch.pth'))  # Replace 'model_path.pth' with the actual model path
model.eval()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Classification and Color Extraction API!"}

def extract_colors(image_path, num_colors=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize GrabCut parameters
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, image.shape[1]-10, image.shape[0]-10)  # Rectangle for GrabCut (change if needed)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Use mask to separate foreground
    image_fg = image * mask2[:, :, np.newaxis]

    # Extract the masked area for color analysis
    idx = np.where(mask2 != 0)
    pixels = image[idx[0], idx[1], :]

    # Apply K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_

    # Count the occurrences of each label
    labels = kmeans.labels_
    counts = np.bincount(labels)

    return colors.astype(int), counts

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
