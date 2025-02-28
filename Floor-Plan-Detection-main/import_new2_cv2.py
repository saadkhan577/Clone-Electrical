import cv2
import numpy as np
import pytesseract

# Point pytesseract to the correct tesseract.exe location
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\haier\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

image_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\Images\example.png"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
text = pytesseract.image_to_string(gray)
print(text)
