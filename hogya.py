import os
import torch
import easyocr  # OCR library for text extraction
import pandas as pd
import re
from tqdm import tqdm
from urllib.request import urlretrieve
from PIL import Image, ImageEnhance
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# Define the entity-unit map with the full and short forms of measurement units
entity_unit_map = {
    'width': {'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd'},
    'depth': {'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd'},
    'height': {'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd'},
    'item_weight': {'gram', 'g', 'kilogram', 'kg', 'microgram', 'μg', 'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'ton'},
    'maximum_weight_recommendation': {'gram', 'g', 'kilogram', 'kg', 'microgram', 'μg', 'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'ton'},
    'voltage': {'volt', 'V', 'millivolt', 'mV', 'kilovolt', 'kV'},
    'wattage': {'watt', 'W', 'kilowatt', 'kW'},
    'item_volume': {'litre', 'L', 'millilitre', 'ml', 'centilitre', 'cl', 'cubic foot', 'cu ft', 'cubic inch', 'cu in', 
                    'cup', 'fluid ounce', 'fl oz', 'gallon', 'imperial gallon', 'quart', 'pint'}
}

# Initialize EasyOCR reader (GPU if available)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Pretrained ResNet model for multi-task learning (for predicting values and units)
class ResNetMultiTaskModel(nn.Module):
    def __init__(self, num_units):
        super(ResNetMultiTaskModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        
        # Regression head for predicting numeric values
        self.regression_head = nn.Linear(2048, 1)  # Outputs a single value (regression task)
        
        # Classification head for predicting units
        self.classification_head = nn.Linear(2048, num_units)  # Outputs one of the possible units

    def forward(self, x):
        x = self.resnet(x)
        value = self.regression_head(x)
        unit = self.classification_head(x)
        return value, unit

# Function to load and preprocess image for Deep Learning model
def preprocess_image_dl(image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Enhance contrast to improve model's performance (optional)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Function to map short unit forms to their full forms
def expand_unit_short_forms(unit):
    unit_mapping = {
        'cm': 'centimetre',
        'mm': 'millimetre',
        'm': 'metre',
        'in': 'inch',
        'ft': 'foot',
        'yd': 'yard',
        'g': 'gram',
        'kg': 'kilogram',
        'mg': 'milligram',
        'μg': 'microgram',
        'oz': 'ounce',
        'lb': 'pound',
        'L': 'litre',
        'ml': 'millilitre',
        'cl': 'centilitre',
        'mV': 'millivolt',
        'kV': 'kilovolt',
        'V': 'volt',
        'W': 'watt',
        'kW': 'kilowatt'
    }
    return unit_mapping.get(unit, unit)  # Returns expanded form if found, else returns original

# Extract text from image using EasyOCR
def extract_text(image_path):
    result = reader.readtext(image_path)
    extracted_text = " ".join([item[1] for item in result])  # Combine OCR results into a single string
    return extracted_text

# Function to extract value and unit based on entity type
def extract_value_and_unit_ocr(extracted_text, entity_name):
    # Use regular expressions to find numeric values
    value_match = re.search(r"(\d+(\.\d+)?)", extracted_text)
    if value_match:
        value = float(value_match.group(0))  # Convert the matched string to a float
    else:
        return None, None  # No valid numeric value found
    
    # Look for the unit in the allowed units for the entity
    allowed_units = entity_unit_map.get(entity_name, [])
    for unit in allowed_units:
        if unit in extracted_text:
            expanded_unit = expand_unit_short_forms(unit)
            return value, expanded_unit
    
    return value, None  # Return value if found, even if the unit is missing

# Function to train the Deep Learning model
def train_resnet_model(dl_model, train_df, num_units, device, epochs=1):
    # Optimizer and loss functions
    optimizer = optim.Adam(dl_model.parameters(), lr=0.001)
    criterion_regression = nn.MSELoss()  # For regression
    criterion_classification = nn.CrossEntropyLoss()  # For classification (unit prediction)

    dl_model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
            image_url = row['image_link']
            entity_value = row['entity_value']
            entity_name = row['entity_name']
            
            # Download and preprocess image
            image_path = download_image(image_url, '/content/ml/images/')
            image = preprocess_image_dl(image_path).to(device)
            
            # Parse value and unit from the data
            value, unit = parse_entity_value(entity_value)
            if value is None or unit is None:
                continue  # Skip invalid rows
            
            # Prepare target tensors
            value_target = torch.tensor([value], dtype=torch.float32).to(device)
            unit_target = torch.tensor([list(entity_unit_map[entity_name]).index(unit)], dtype=torch.long).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predicted_value, predicted_unit_logits = dl_model(image)
            
            # Compute loss
            loss_value = criterion_regression(predicted_value, value_target)
            loss_unit = criterion_classification(predicted_unit_logits, unit_target)
            loss = loss_value + loss_unit
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_df)}")

    # Save the trained model
    torch.save(dl_model.state_dict(), './resnet_multitask.pth')
    print("Model saved!")

# Function to download an image from a URL
def download_image(image_url, image_folder):
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    image_filename = os.path.join(image_folder, os.path.basename(image_url))
    urlretrieve(image_url, image_filename)
    return image_filename

# Function to process images with OCR first, and fallback to DL model if necessary
def process_images_with_fallback(test_df, image_folder, dl_model, device, limit=51):
    test_df = test_df.head(limit)
    test_df['prediction'] = ""
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_url = row['image_link']
        entity_name = row['entity_name']
        
        # Download and preprocess image
        image_path = download_image(image_url, image_folder)
        
        # First attempt OCR extraction
        extracted_text = extract_text(image_path)
        value, unit = extract_value_and_unit_ocr(extracted_text, entity_name)
        
        # If OCR fails to extract valid data, fall back to Deep Learning model
        if value is None or unit is None:
            print(f"Falling back to DL model for {image_path}")
            image_dl = preprocess_image_dl(image_path).to(device)
            dl_model.eval()
            with torch.no_grad():
                predicted_value, predicted_unit_logits = dl_model(image_dl)
                predicted_value = predicted_value.item()
                
                predicted_unit = predicted_unit_logits.argmax(dim=1).item()
                units_list = list(entity_unit_map.get(entity_name, []))
                if predicted_unit >= len(units_list):
                    unit = "unknown"  # Handle invalid unit case
                else:
                    unit = units_list[predicted_unit]  # Map to correct unit

                value = predicted_value
        
        test_df.at[idx, 'prediction'] = f"{value} {unit}"
    
    return test_df

# Main function to combine OCR and DL model for prediction
def main_combined_model():
    DATASET_FOLDER = '/content/drive/MyDrive/ml'
    IMAGE_FOLDER = '/content/drive/MyDrive/ml/images'
    TEST_CSV = os.path.join(DATASET_FOLDER, 'test.csv')
    OUTPUT_CSV = os.path.join(DATASET_FOLDER, 'test_out.csv')
    
    test_df = pd.read_csv(TEST_CSV)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_units = max(len(v) for v in entity_unit_map.values())  # Maximum number of units across all entities
    dl_model = ResNetMultiTaskModel(num_units).to(device)
    
    test_df = process_images_with_fallback(test_df, IMAGE_FOLDER, dl_model, device, limit=51)
    
    test_df[['index', 'prediction']].to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main_combined_model()
