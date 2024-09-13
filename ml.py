import requests
from io import BytesIO
from PIL import Image
import pytesseract
import pandas as pd
import re
from sklearn.linear_model import LinearRegression

# Categories mapping
dimension_names = ['height', 'width', 'depth']
weight_names = ['item_weight', 'maximum_weight_recommendation']
volume_names = ['item_volume']

# Function to fetch image from URL
def fetch_image_from_url(url):
    """Fetches image from URL and returns it as a PIL image."""
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error fetching image from URL {url}: {e}")
        return None

# Function to extract text from image using OCR
def extract_text_from_image(image):
    """Extracts text from a PIL image using OCR."""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

# Function to extract numerical value and unit from text
def extract_value_and_unit(text):
    """Extracts numerical value and unit from text."""
    match = re.search(r'(\d+(\.\d+)?)\s*(kg|g|lb|oz|cm|m|mm|in|ft|yard|cup|milligram|litre|liter)', text, re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(3).lower()
        return value, unit
    return None, None

# Function to process images and extract dimensions
def process_images_and_extract_dimensions(image_urls):
    """Fetches images from URLs and extracts dimensions using OCR."""
    values = []
    units = []
    for url in image_urls:
        print(f"Processing URL: {url}")  # Debugging statement
        img = fetch_image_from_url(url)
        if img:
            text = extract_text_from_image(img)
            value, unit = extract_value_and_unit(text)
            values.append(value)
            units.append(unit)
        else:
            values.append(None)
            units.append(None)
    return values, units

# Load train and test data
train_df = pd.read_csv('/workspaces/python/amazon_ml/student_resource_3/dataset/train.csv')
test_df = pd.read_csv('/workspaces/python/amazon_ml/student_resource_3/dataset/test.csv')

# Optional: Take only the first 20 rows for testing (comment out if not needed)
train_df = train_df.head(20)
test_df = test_df.head(20)

# Print a few rows of training data for inspection
print("Training Data Sample:")
print(train_df.head())

# Extract image URLs and dimensions from train data
train_image_urls = train_df['image_link'].tolist()
train_entity_names = train_df['entity_name'].tolist()
train_entity_values = train_df['entity_value'].tolist()

# Extract image URLs from test data
test_image_urls = test_df['image_link'].tolist()

# Process training images to extract values and units
train_values, train_units = process_images_and_extract_dimensions(train_image_urls)

# Map entity names to volumes, weights, and dimensions for training data
volume_values = []
weight_values = []
dimension_values = []

for name, value, unit in zip(train_entity_names, train_values, train_units):
    if name in weight_names:
        weight_values.append(value)
        volume_values.append(None)
        dimension_values.append(None)
    elif name in volume_names:
        volume_values.append(value)
        weight_values.append(None)
        dimension_values.append(None)
    elif name in dimension_names:
        dimension_values.append(value)
        weight_values.append(None)
        volume_values.append(None)
    else:
        weight_values.append(None)
        volume_values.append(None)
        dimension_values.append(None)

# Print some of the extracted values
print("Extracted Values from Training Images:")
print(f"Volumes: {volume_values}")
print(f"Weights: {weight_values}")
print(f"Dimensions: {dimension_values}")

# Create DataFrame for training data with extracted values
train_data = pd.DataFrame({
    'image_link': train_image_urls,
    'entity_name': train_entity_names,
    'entity_value': train_entity_values,
    'extracted_value': train_values,
    'extracted_unit': train_units
})

# Handle NaN values in 'extracted_value'
train_data['extracted_value'] = train_data['extracted_value'].fillna(0)

# Separate training data for volume, weight, and dimensions
volume_train_data = train_data[train_data['entity_name'].isin(volume_names)]
weight_train_data = train_data[train_data['entity_name'].isin(weight_names)]
dimension_train_data = train_data[train_data['entity_name'].isin(dimension_names)]

# Features and labels for volume, weight, and dimensions
X_train_volume = volume_train_data['extracted_value'].values.reshape(-1, 1)
y_train_volume = volume_train_data['entity_value'].str.extract(r'(\d+(\.\d+)?)').astype(float)

X_train_weight = weight_train_data['extracted_value'].values.reshape(-1, 1)
y_train_weight = weight_train_data['entity_value'].str.extract(r'(\d+(\.\d+)?)').astype(float)

X_train_dimension = dimension_train_data['extracted_value'].values.reshape(-1, 1)
y_train_dimension = dimension_train_data['entity_value'].str.extract(r'(\d+(\.\d+)?)').astype(float)

# Drop rows with NaN values in the features or target arrays
valid_volume_idx = ~y_train_volume.isna().any(axis=1) & ~pd.isna(X_train_volume).any(axis=1)
X_train_volume = X_train_volume[valid_volume_idx]
y_train_volume = y_train_volume[valid_volume_idx]

valid_weight_idx = ~y_train_weight.isna().any(axis=1) & ~pd.isna(X_train_weight).any(axis=1)
X_train_weight = X_train_weight[valid_weight_idx]
y_train_weight = y_train_weight[valid_weight_idx]

valid_dimension_idx = ~y_train_dimension.isna().any(axis=1) & ~pd.isna(X_train_dimension).any(axis=1)
X_train_dimension = X_train_dimension[valid_dimension_idx]
y_train_dimension = y_train_dimension[valid_dimension_idx]

# Train regression models only if there is valid data
models = {}

# Train volume model if there is valid data
if len(X_train_volume) > 0 and len(y_train_volume) > 0:
    model_volume = LinearRegression()
    model_volume.fit(X_train_volume, y_train_volume)
    models['volume'] = model_volume
    print("Volume model trained successfully.")
else:
    print("Skipping volume model training due to insufficient data.")

# Train weight model if there is valid data
if len(X_train_weight) > 0 and len(y_train_weight) > 0:
    model_weight = LinearRegression()
    model_weight.fit(X_train_weight, y_train_weight)
    models['weight'] = model_weight
    print("Weight model trained successfully.")
else:
    print("Skipping weight model training due to insufficient data.")

# Train dimension model if there is valid data
if len(X_train_dimension) > 0 and len(y_train_dimension) > 0:
    model_dimension = LinearRegression()
    model_dimension.fit(X_train_dimension, y_train_dimension)
    models['dimension'] = model_dimension
    print("Dimension model trained successfully.")
else:
    print("Skipping dimension model training due to insufficient data.")

# Extract values and units from test images
test_values, test_units = process_images_and_extract_dimensions(test_image_urls)

# Print some of the extracted values from test images
print("Extracted Values from Test Images:")
print(f"Values: {test_values}")

# Create DataFrame for test data with extracted dimensions
test_data = pd.DataFrame({
    'index': test_df.index,  # Ensure we use the DataFrame index for output
    'extracted_value': test_values,
    'extracted_unit': test_units
})

# Handle NaN values in 'extracted_value'
test_data['extracted_value'] = test_data['extracted_value'].fillna(0)

# Convert test values to features
X_test = test_data['extracted_value'].values.reshape(-1, 1)

# Make predictions only for the models that were successfully trained
def format_prediction(prediction):
    """Format prediction to a float and append the unit."""
    try:
        if len(prediction) == 1:
            return f"{float(prediction[0]):.2f} unit"
        return ""
    except Exception as e:
        print(f"Error formatting prediction: {e}")
        return ""

# Apply predictions for each available model
predictions_volume = [format_prediction(pred) for pred in models['volume'].predict(X_test)] if 'volume' in models else ["" for _ in range(len(X_test))]
predictions_weight = [format_prediction(pred) for pred in models['weight'].predict(X_test)] if 'weight' in models else ["" for _ in range(len(X_test))]
predictions_dimension = [format_prediction(pred) for pred in models['dimension'].predict(X_test)] if 'dimension' in models else ["" for _ in range(len(X_test))]

# Create output DataFrame
output_df = pd.DataFrame({
    'index': test_data['index'],
    'predicted_volume': predictions_volume,
    'predicted_weight': predictions_weight,
    'predicted_dimension': predictions_dimension,
})

# Save output to CSV
output_df.to_csv('/workspaces/python/amazon_ml/student_resource_3/output.csv', index=False)
print("Output saved to output.csv")
