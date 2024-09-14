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

# Define the entity-unit map with short forms of units
entity_unit_map = {
    'width': {'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd'},
    'depth': {'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd'},
    'height': {'centimetre', 'cm', 'foot', 'ft', 'inch', 'in', 'metre', 'm', 'millimetre', 'mm', 'yard', 'yd'},
    'item_weight': {'gram', 'g', 'kilogram', 'kg', 'microgram', 'ug', 'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'ton'},
    'maximum_weight_recommendation': {'gram', 'g', 'kilogram', 'kg', 'microgram', 'ug', 'milligram', 'mg', 'ounce', 'oz', 'pound', 'lb', 'ton'},
    'voltage': {'volt', 'V', 'millivolt', 'mV', 'kilovolt', 'kV'},
    'wattage': {'watt', 'W', 'kilowatt', 'kW'},
    'item_volume': {'litre', 'L', 'ml', 'millilitre', 'centilitre', 'cL', 'cubic foot', 'cu ft', 'cubic inch', 'cu in', 'cup', 'fluid ounce', 'oz', 'gallon', 'imperial gallon', 'quart', 'pint'}
}

# Initialize EasyOCR reader (GPU if available)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Pretrained ResNet model for multi-task learning (Deep Learning Approach)
class ResNetMultiTaskModel(nn.Module):
    def __init__(self, num_units):
        super(ResNetMultiTaskModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final layer
        
        # Regression head for predicting numeric values
        self.regression_head = nn.Linear(2048, 1)  # Output 1 value (regression)
        
        # Classification head for predicting units
        self.classification_head = nn.Linear(2048, num_units)  # Output number of units

    def forward(self, x):
        x = self.resnet(x)
        value = self.regression_head(x)
        unit = self.classification_head(x)
        return value, unit

# Function to load and preprocess image for Deep Learning model
def preprocess_image_dl(image_path):
    image = Image.open(image_path).convert('RGB')
    
    # Enhance contrast (optional but may improve DL accuracy)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Download the image from the link
def download_image(image_link, image_folder):
    image_name = os.path.basename(image_link)
    image_path = os.path.join(image_folder, image_name)
    if not os.path.exists(image_path):
        urlretrieve(image_link, image_path)
    return image_path

# Parse the entity_value into numeric value and unit
def parse_entity_value(entity_value):
    # Handle cases where entity_value might contain ranges (e.g., '15 kilogram to 20 kilogram')
    try:
        # Use regular expressions to extract only the first valid number and unit
        value_match = re.search(r"(\d+(\.\d+)?)", entity_value)  # Find the first numeric value
        unit_match = re.search(r"\b[a-zA-Z]+\b", entity_value)  # Find the first unit string
        
        if value_match and unit_match:
            value = float(value_match.group(0))  # Extract numeric value
            unit = unit_match.group(0).strip()   # Extract the first unit and strip any surrounding whitespace
            return value, unit
        else:
            return None, None
    except Exception as e:
        print(f"Error parsing entity_value: {entity_value}, Error: {e}")
        return None, None

# Train the Deep Learning model (limit training to 50 images)
def train_resnet_model(dl_model, train_df, num_units, device, epochs=50):
    # Limit training data to the first 50 rows
    train_df = train_df.head(50)  # Train with only 50 images

    # Optimizer and loss functions
    optimizer = optim.Adam(dl_model.parameters(), lr=0.001)
    criterion_regression = nn.MSELoss()  # Mean Squared Error for regression
    criterion_classification = nn.CrossEntropyLoss()  # Cross Entropy Loss for classification

    # Set model to training mode
    dl_model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
            image_url = row['image_link']
            entity_value = row['entity_value']
            entity_name = row['entity_name']
            
            # Download and preprocess image
            image_path = download_image(image_url, '/content/ml/images')
            image = preprocess_image_dl(image_path).to(device)
            
            # Parse value and unit
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
            
            # Calculate loss
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

# Process the images using OCR first, fall back to DL model if OCR fails
def process_images_with_fallback(test_df, image_folder, dl_model, device, limit=50):
    test_df = test_df.head(limit)  # Limit the dataframe to the first 50 images
    test_df['prediction'] = ""
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_url = row['image_link']
        entity_name = row['entity_name']
        
        # Download image and preprocess
        image_path = download_image(image_url, image_folder)
        
        # First, attempt OCR extraction
        extracted_text = extract_text(image_path)
        value, unit = extract_value_and_unit_ocr(extracted_text, entity_name)
        
        # If OCR did not find valid data, fall back to Deep Learning model
        if value is None or unit is None:
            print(f"Falling back to DL model for {image_path}")
            # Preprocess image for DL model
            image_dl = preprocess_image_dl(image_path).to(device)
            dl_model.eval()
            with torch.no_grad():
                predicted_value, predicted_unit_logits = dl_model(image_dl)
                predicted_value = predicted_value.item()
                predicted_unit = predicted_unit_logits.argmax(dim=1).item()
                unit = list(entity_unit_map[entity_name])[predicted_unit]  # Map to correct unit
                value = predicted_value
        
        # Save prediction
        test_df.at[idx, 'prediction'] = f"{value} {unit}"
    
    return test_df

# Main function for training and combined model (OCR + DL fallback)
def main_train_and_test():
    # Paths
    DATASET_FOLDER = '/content/ml'
    IMAGE_FOLDER = '/content/ml/images'
    TRAIN_CSV = os.path.join(DATASET_FOLDER, 'train.csv')
    TEST_CSV = os.path.join(DATASET_FOLDER, 'test.csv')
    OUTPUT_CSV = os.path.join(DATASET_FOLDER, 'test_out.csv')
    
    # Load train data for training
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Load Deep Learning model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_units = max(len(v) for v in entity_unit_map.values())  # Calculate max number of units
    dl_model = ResNetMultiTaskModel(num_units).to(device)
    
    # Train the model using train.csv (limit to 50 images)
    print("Training ResNet Multi-task model with 50 images from train.csv...")
    train_resnet_model(dl_model, train_df, num_units, device, epochs=50)
    
    # Load test data
    test_df = pd.read_csv(TEST_CSV)
    
    # Process images and generate predictions (limit to 50 images)
    test_df = process_images_with_fallback(test_df, IMAGE_FOLDER, dl_model, device, limit=50)
    
    # Save results
    test_df[['index', 'prediction']].to_csv(OUTPUT_CSV, index=False)
    print(f"Predictions saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    # Run the training and prediction pipeline
    print("Running training and combined OCR + Deep Learning model with 50 images...")
    main_train_and_test()
