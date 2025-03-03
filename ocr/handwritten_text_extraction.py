import cv2
import pytesseract

def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Apply thresholding to get a binary image
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary_image

def extract_handwritten_text(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Use pytesseract to extract text from the preprocessed image
    extracted_text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')
    
    return extracted_text
