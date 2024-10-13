import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np
import re
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('handwritten.h5')

# Set the path for Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Function to convert PDF to images
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return images


# Function to filter red marks from an image
def filter_red_marks(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_marks = cv2.bitwise_and(image, image, mask=red_mask)
    gray_marks = cv2.cvtColor(red_marks, cv2.COLOR_BGR2GRAY)
    return gray_marks


# Function to predict marks using the model
def predict_marks(image):
    # Preprocess the image for the model
    img = cv2.resize(image, (28, 28))  # Resize to 28x28
    img = img.astype('float32') / 255.0  # Normalize
    img = np.invert(img)  # Invert colors
    img = img.reshape(1, 28, 28)  # Reshape for the model input
    prediction = model.predict(img)
    return np.argmax(prediction)


# Function to extract and return marks
def extract_marks(image):
    # Extract each digit from the image and predict
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        digit_img = image[y:y + h, x:x + w]  # Extract the digit
        predicted_mark = predict_marks(digit_img)
        marks.append(predicted_mark)

    return marks


# Main function
def process_exam_copy(input_path):
    total_marks = 0
    all_marks = []  # To store marks from each page or image

    if input_path.lower().endswith('.pdf'):
        images = pdf_to_images(input_path)
    else:
        images = [cv2.imread(input_path)]

    for i, img in enumerate(images):
        red_marks_img = filter_red_marks(img)
        marks = extract_marks(red_marks_img)

        # Sum the predicted marks
        page_total = sum(marks)
        total_marks += page_total

        # Store and print marks for the current page/image
        all_marks.append(marks)
        print(f"Marks found on page {i + 1}: {marks}")

    print(f"Total Marks: {total_marks}")
    return total_marks


# Example usage
input_file = r'C:\Users\shiva\Downloads\SDL\Document_46_1.pdf'
total = process_exam_copy(input_file)
