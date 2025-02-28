import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import os

#---------------------
# 1. Load the Image
#---------------------
image_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\Images\example.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Read image (BGR format)
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Could not load the image. Check the file path or integrity.")

#---------------------
# 2. Preprocessing
#---------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)  # Detect edges

#---------------------
# 3. Contour Detection
#---------------------
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copy of original for drawing
output = image.copy()

# Draw all contours in green
cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

#---------------------
# 4. Tesseract OCR
#---------------------
# Point Tesseract to the correct install location
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\haier\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Dictionary of keywords -> colors (B, G, R)
# Feel free to add or adjust keywords/labels
label_colors = {
    "KITCHEN": (255, 0, 0),   # Blue
    "BATH":    (0, 0, 255),   # Red
    "BED":     (0, 255, 255), # Yellow
    "ROOM":    (255, 0, 255), # Magenta
    "WALL":    (0, 255, 0),   # Green
}

# We'll store any recognized text here
detected_labels = {}

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w < 20 or h < 20:
        # Skip tiny contours
        continue

    roi = gray[y:y+h, x:x+w]

    # OCR on this region
    text = pytesseract.image_to_string(roi, config="--psm 6").strip().upper()

    # If Tesseract found something, draw a red box + label
    if len(text) > 2:
        # Save bounding box for debugging
        detected_labels[text] = (x, y, w, h)

        # Default: red rectangle
        rect_color = (0, 0, 255)

        # If text contains a known keyword, use that color
        for keyword, color in label_colors.items():
            if keyword in text:
                rect_color = color
                break

        # Draw bounding box
        cv2.rectangle(output, (x, y), (x+w, y+h), rect_color, 2)

        # Put recognized text above box
        cv2.putText(
            output, 
            text, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            rect_color, 
            2
        )

#---------------------
# 5. Show Final Result in a Resizable Window
#---------------------
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
cv2.imshow("Detected Objects", output)

# Optionally resize the window to a comfortable size (e.g., 1280×720)
cv2.resizeWindow("Detected Objects", 1280, 720)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Print recognized text and bounding boxes
print("Detected Labels & Positions:")
for label, bbox in detected_labels.items():
    print(f"{label} -> {bbox}")
