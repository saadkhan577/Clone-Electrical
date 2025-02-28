import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.FloorplanToBlenderLib import *
from utils.loaders import FloorplanSVG, DictToTensor, Compose, RotateNTurns
from utils.post_prosessing import split_prediction, get_polygons
from model import get_model

# Load the model
def load_model(model_path, device):
    model = get_model('hg_furukawa_original', 51)
    n_classes = 44
    model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    model.to(device)
    return model

# Process input image
def preprocess_image(img_path, device):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = 2 * (img / 255.0) - 1
    img = np.moveaxis(img, -1, 0)
    img = torch.tensor([img.astype(np.float32)]).to(device)
    return img

# Perform inference
def predict_floorplan(model, img, device, split):
    img_size = (img.shape[2], img.shape[3])
    prediction = torch.zeros([4, 44, *img_size])
    rot = RotateNTurns()
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    
    with torch.no_grad():
        for i, (forward, back) in enumerate(rotations):
            rot_img = rot(img, 'tensor', forward)
            pred = model(rot_img)
            pred = rot(pred, 'tensor', back)
            pred = rot(pred, 'points', back)
            pred = F.interpolate(pred, size=img_size, mode='bilinear', align_corners=True)
            prediction[i] = pred[0]
    
    prediction = torch.mean(prediction, 0, True)
    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    return get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

# Visualize results
def visualize_results(image_path, polygons, room_types):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    for poly, room in zip(polygons, room_types):
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (255, 0, 0), 2)
        x, y = np.mean(pts, axis=0)[0]
        plt.text(x, y, room, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    plt.show()

# Main execution
def main(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    img = preprocess_image(image_path, device)
    split = [21, 12, 11]
    polygons, types, room_polygons, room_types = predict_floorplan(model, img, device, split)
    visualize_results(image_path, room_polygons, room_types)

# Run the model on your floor plan
image_path = "Images/example.png"  # Change this to your uploaded file path
model_path ="Floor-Plan-Detection-main/model/__pycache__/model_1427.cpython-313.pyc"  # Change if using a different model checkpoint
main(image_path, model_path)
