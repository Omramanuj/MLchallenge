

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
from PIL import Image

# Define the allowed units for each entity type
ALLOWED_UNITS = {
    'item_weight': ['gram', 'kilogram', 'ounce', 'pound'],
    'item_length': ['millimetre', 'centimetre', 'metre', 'inch', 'foot', 'yard'],
    'item_width': ['millimetre', 'centimetre', 'metre', 'inch', 'foot', 'yard'],
    'item_height': ['millimetre', 'centimetre', 'metre', 'inch', 'foot', 'yard'],
    'item_voltage': ['volt', 'kilovolt', 'millivolt'],
    'item_wattage': ['watt', 'kilowatt'],
    # Add more entity types and their allowed units as needed
}

def process_image(image_link):
    try:
        response = requests.get(image_link, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')  # Ensure 3 color channels
            img = img.resize((224, 224))  # Resize to a fixed size
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            return np.expand_dims(img_array, axis=0)  # Add batch dimension
        else:
            print(f"Failed to download {image_link}: HTTP status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return None

def format_prediction(value, entity_name):
    if value <= 0:
        return ""
    
    allowed_units = ALLOWED_UNITS.get(entity_name, [])
    if not allowed_units:
        return ""
    
    # Choose an appropriate unit based on the value and entity type
    if entity_name in ['item_weight', 'item_length', 'item_width', 'item_height']:
        if value < 1:
            unit = allowed_units[0]  # Smallest unit (e.g., gram, millimetre)
        elif value > 1000:
            unit = allowed_units[1]  # Largest unit (e.g., kilogram, metre)
            value /= 1000  # Convert to larger unit
        else:
            unit = allowed_units[0]  # Use the smallest unit as default
    elif entity_name == 'item_voltage':
        if value >= 1000:
            unit = 'kilovolt'
            value /= 1000
        elif value < 1:
            unit = 'millivolt'
            value *= 1000
        else:
            unit = 'volt'
    elif entity_name == 'item_wattage':
        if value >= 1000:
            unit = 'kilowatt'
            value /= 1000
        else:
            unit = 'watt'
    else:
        unit = allowed_units[0]  # Default to first unit if entity type is unknown
    
    return f"{value:.2f} {unit}"

# Load the trained model
model = load_model('path_to_your_model.h5')

# Load the test data
test_df = pd.read_csv('dataset/test.csv')

# Make predictions
predictions = []
for index, row in test_df.iterrows():
    image_link = row['image_link']
    entity_name = row['entity_name']
    
    # Process the image
    img_array = process_image(image_link)
    
    if img_array is not None:
        # Make prediction
        prediction = model.predict(img_array)[0][0]
        
        # Format the prediction
        formatted_prediction = format_prediction(prediction, entity_name)
    else:
        formatted_prediction = ""  # Empty string for failed image processing
    
    predictions.append({
        'index': index,
        'prediction': formatted_prediction
    })

# Create the output DataFrame
output_df = pd.DataFrame(predictions)

# Save the output to CSV
output_df.to_csv('test_out.csv', index=False)

print("Predictions saved to test_out.csv")