import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 1) Add the repo to Python path (adjust path to where you cloned the repo)
sys.path.append(r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main")

from model import get_model
from utils.post_prosessing import split_prediction, get_polygons, split_validation
from utils.plotting import polygons_to_image

# Example classes from the CubiCasa model
room_classes = [
    "Wall", "Kitchen", "Living Room", "Bed Room", "Bath",
    "Entry", "Storage", "Garage", "Undefined"
]
icon_classes = [
    "No Icon", "Window", "Door", "Closet", "Electrical Appliance",
    "Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"
]

# Helper function for rotating a tensor by multiples of 90 degrees
def rotate_tensor(tensor, rotation):
    """
    Rotates the input tensor by rotation*90 degrees.
    
    Args:
        tensor (Tensor): Input tensor of shape (B, C, H, W)
        rotation (int): Number of 90° rotations (can be negative)
    
    Returns:
        Tensor: Rotated tensor
    """
    k = rotation % 4  # Normalize to [0,3]
    return torch.rot90(tensor, k, dims=(2, 3))

# 2) Load the pre-trained model
def load_pretrained_model(model_path, device):
    model = get_model('hg_furukawa_original', 51)  # Hourglass model
    n_classes = 44  # Typically 44 total output channels
    model.conv4_ = torch.nn.Conv2d(256, n_classes, kernel_size=1)
    model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model.to(device)
    return model

# 3) Preprocess your floor plan image
def preprocess_image(img_path, device):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize to range [-1,1]
    img = 2.0 * (img / 255.0) - 1.0
    # Change shape from (H, W, 3) -> (3, H, W)
    img = np.moveaxis(img, -1, 0)
    # Convert to Torch tensor, add batch dimension
    img_tensor = torch.from_numpy(np.array([img.astype(np.float32)])).to(device)
    return img_tensor

# 4) Perform inference and segmentation
def predict_segmentation(model, img_tensor, device, split=[21, 12, 11]):
    # Typically the model is run on 4 rotations for better results
    rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
    batch, channels, height, width = img_tensor.shape
    n_classes = 44
    prediction = torch.zeros([len(rotations), n_classes, height, width], device=device)

    for i, (fwd, bwd) in enumerate(rotations):
        # Rotate input, predict, then rotate output back
        rotated_input = rotate_tensor(img_tensor, fwd)
        pred = model(rotated_input)
        pred = rotate_tensor(pred, bwd)
        # Upsample to original size
        pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
        prediction[i] = pred[0]

    # Average the predictions from all rotations
    prediction = torch.mean(prediction, dim=0, keepdim=True)

    # Split into heatmaps, room seg, icon seg
    heatmaps, rooms, icons = split_prediction(prediction, (height, width), split)
    # Convert raw segmentation to polygons or a color-coded mask
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

    return polygons, types, room_polygons, room_types

# 5) Visualize color-coded segmentation
def visualize_segmentation(img_path, polygons, types, room_polygons, room_types, n_rooms=12):
    # Re-load original image for display
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # Convert polygons to a labeled mask
    # polygons_to_image returns separate masks for rooms and icons
    pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types,
                                                   orig_img.shape[0], orig_img.shape[1])

    # Plot using matplotlib
    plt.figure(figsize=(14, 6))

    # Show color-coded room segmentation
    plt.subplot(1, 2, 1)
    plt.imshow(pol_room_seg, vmin=0, vmax=n_rooms - 0.1)
    plt.title("Room Segmentation")
    plt.axis('off')

    # Show icon segmentation
    plt.subplot(1, 2, 2)
    plt.imshow(pol_icon_seg, vmin=0, vmax=11.9)  # if 12 icon classes
    plt.title("Icon Segmentation")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 6) Main script usage
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths to your model and floor plan
    model_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\model\model_1427.pth"
    floorplan_img_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\Images\example.png"

    model = load_pretrained_model(model_path, device)
    img_tensor = preprocess_image(floorplan_img_path, device)
    polygons, types, room_polygons, room_types = predict_segmentation(model, img_tensor, device)

    visualize_segmentation(floorplan_img_path, polygons, types, room_polygons, room_types)

if __name__ == "__main__":
    main()
