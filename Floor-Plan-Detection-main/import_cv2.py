import cv2
import os

# Set the correct path to your image
image_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\Images\example.png"

# Check if file exists
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: File '{image_path}' not found!")

# Load the image
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    raise ValueError(f"Error: Could not read image at {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150)

# Display the results
cv2.imshow("Edges Detected", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
