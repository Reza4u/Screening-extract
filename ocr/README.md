# OCR for Handwritten Text Extraction

This repository contains scripts for extracting handwritten text from images using OCR and training a custom OCR model for handwritten text recognition.

## Installation

To use the scripts in this repository, you need to install the required libraries. You can install them using `pip`:

```bash
pip install opencv-python pytesseract tensorflow
```

## Usage

### Extract Handwritten Text

The `handwritten_text_extraction.py` script can be used to extract handwritten text from images.

#### Example

```python
from handwritten_text_extraction import extract_handwritten_text

image_path = 'path/to/your/handwritten_image.jpg'
extracted_text = extract_handwritten_text(image_path)
print(extracted_text)
```

### Train Custom OCR Model

The `train_custom_model.py` script can be used to train a custom OCR model for handwritten text recognition.

#### Example

```python
from train_custom_model import train_model, load_data, preprocess_data, create_model, evaluate_model

data_dir = 'path/to/your/data'
images, labels = load_data(data_dir)
images, labels = preprocess_data(images, labels)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
input_shape = (128, 128, 1)
num_classes = len(set(labels))
model = create_model(input_shape, num_classes)
train_model(model, train_images, train_labels, test_images, test_labels)
evaluate_model(model, test_images, test_labels)
```

## License

This project is licensed under the MIT License.
