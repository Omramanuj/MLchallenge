# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import requests
# from io import BytesIO
# from PIL import Image
# from tqdm import tqdm

# # Define the allowed units for each entity type
# entity_unit_map = {
#     'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'voltage': {'kilovolt', 'millivolt', 'volt'},
#     'wattage': {'kilowatt', 'watt'},
#     'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
# }

# def process_image(image_link):
#     try:
#         response = requests.get(image_link, stream=True, timeout=10)
#         if response.status_code == 200:
#             img = Image.open(BytesIO(response.content))
#             img = img.convert('RGB')  # Ensure 3 color channels
#             img = img.resize((224, 224))  # Resize to a fixed size
#             img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#             return np.expand_dims(img_array, axis=0)  # Add batch dimension
#         else:
#             print(f"Failed to download {image_link}: HTTP status {response.status_code}")
#             return None
#     except Exception as e:
#         print(f"Error processing {image_link}: {e}")
#         return None

# def format_prediction(value, entity_name):
#     if value <= 0:
#         return ""
    
#     allowed_units = list(entity_unit_map.get(entity_name, set()))
#     if not allowed_units:
#         return ""
    
#     if entity_name in ['width', 'depth', 'height']:
#         if value < 10:
#             unit = 'millimetre'
#         elif value < 1000:
#             unit = 'centimetre'
#             value /= 10
#         else:
#             unit = 'metre'
#             value /= 1000
#     elif entity_name in ['item_weight', 'maximum_weight_recommendation']:
#         if value < 1000:
#             unit = 'gram'
#         elif value < 1000000:
#             unit = 'kilogram'
#             value /= 1000
#         else:
#             unit = 'ton'
#             value /= 1000000
#     elif entity_name == 'voltage':
#         if value >= 1000:
#             unit = 'kilovolt'
#             value /= 1000
#         elif value < 1:
#             unit = 'millivolt'
#             value *= 1000
#         else:
#             unit = 'volt'
#     elif entity_name == 'wattage':
#         if value >= 1000:
#             unit = 'kilowatt'
#             value /= 1000
#         else:
#             unit = 'watt'
#     elif entity_name == 'item_volume':
#         if value < 1000:
#             unit = 'millilitre'
#         elif value < 1000000:
#             unit = 'litre'
#             value /= 1000
#         else:
#             unit = 'gallon'
#             value /= 3785.41  # Convert mL to gallons
#     else:
#         unit = allowed_units[0]  # Default to first unit if entity type is unknown
    
#     return f"{value:.2f} {unit}"

# # Load the trained model
# model = load_model("C:/Users/Ajay/Desktop/MLchallenge/cnn_regression_model.h5")

# # Load the test data
# test_df = pd.read_csv("C:/Users/Ajay/Desktop/MLchallenge/test.csv")

# # Make predictions
# predictions = []
# for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
#     image_link = row['image_link']
#     entity_name = row['entity_name']
    
#     # Process the image
#     img_array = process_image(image_link)
    
#     if img_array is not None:
#         # Make prediction
#         prediction = model.predict(img_array)[0][0]
        
#         # Format the prediction
#         formatted_prediction = format_prediction(prediction, entity_name)
#     else:
#         formatted_prediction = ""  # Empty string for failed image processing
    
#     predictions.append({
#         'index': index,
#         'prediction': formatted_prediction
#     })

# # Create the output DataFrame
# output_df = pd.DataFrame(predictions)

# # Save the output to CSV
# output_df.to_csv('test_out.csv', index=False)

# print("Predictions saved to test_out.csv")

# # Run sanity check
# import subprocess
# subprocess.run(["python", "src/sanity.py", "test_out.csv"])




# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import requests
# from io import BytesIO
# from PIL import Image
# from tqdm.auto import tqdm
# import concurrent.futures

# # Define the allowed units for each entity type (unchanged)
# entity_unit_map = {
#     'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'voltage': {'kilovolt', 'millivolt', 'volt'},
#     'wattage': {'kilowatt', 'watt'},
#     'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
# }

# def process_image(image_link):
#     try:
#         response = requests.get(image_link, stream=True, timeout=10)
#         if response.status_code == 200:
#             img = Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224))
#             return np.array(img) / 255.0
#         else:
#             return None
#     except Exception:
#         return None

# def format_prediction(value, entity_name):
#     # (Keep your existing format_prediction function here)
#     pass

# def process_batch(batch_data):
#     global model  # Use the globally loaded model
#     images, indices, entity_names = batch_data
    
#     if not images:
#         return []
    
#     batch_predictions = model.predict(np.array(images), verbose=0)
    
#     results = []
#     for i, index in enumerate(indices):
#         prediction = batch_predictions[i][0]
#         formatted_prediction = format_prediction(prediction, entity_names[i])
#         results.append({
#             'index': index,
#             'prediction': formatted_prediction
#         })
    
#     return results

# # Load the trained model
# model = load_model("C:/Users/OM/Desktop/Amazon ML School/cnn_regression_model.h5")

# # Load the test data
# test_df = pd.read_csv("C:/Users/OM/Desktop/Amazon ML School/dataset_ml_model/student_resource 3/dataset/test.csv")

# # Prepare batches
# batch_size = 64
# num_workers = 4  # Adjust based on your system's capabilities

# def prepare_batches(df, batch_size):
#     for start in range(0, len(df), batch_size):
#         end = min(start + batch_size, len(df))
#         batch = df.iloc[start:end]
#         images = []
#         indices = []
#         entity_names = []
#         for _, row in batch.iterrows():
#             img = process_image(row['image_link'])
#             if img is not None:
#                 images.append(img)
#                 indices.append(row.name)
#                 entity_names.append(row['entity_name'])
#         yield images, indices, entity_names

# # Process batches in parallel
# all_predictions = []
# with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#     future_to_batch = {executor.submit(process_batch, batch_data): batch_data 
#                        for batch_data in tqdm(prepare_batches(test_df, batch_size), 
#                                               total=len(test_df)//batch_size + 1, 
#                                               desc="Processing batches")}
    
#     for future in tqdm(concurrent.futures.as_completed(future_to_batch), 
#                        total=len(future_to_batch), 
#                        desc="Collecting results"):
#         all_predictions.extend(future.result())

# # Create the output DataFrame
# output_df = pd.DataFrame(all_predictions)

# # Save the output to CSV
# output_df.to_csv('test_out.csv', index=False)

# print("Predictions saved to test_out.csv")

# # Run sanity check
# import subprocess
# subprocess.run(["python", "src/sanity.py", "test_out.csv"])




# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import requests
# from io import BytesIO
# from PIL import Image
# from tqdm import tqdm

# # Define the allowed units for each entity type (unchanged)
# entity_unit_map = {
#     'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'voltage': {'kilovolt', 'millivolt', 'volt'},
#     'wattage': {'kilowatt', 'watt'},
#     'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
# }

# def process_image(image_link):
#     try:
#         response = requests.get(image_link, stream=True, timeout=10)
#         if response.status_code == 200:
#             img = Image.open(BytesIO(response.content))
#             img = img.convert('RGB')  # Ensure 3 color channels
#             img = img.resize((224, 224))  # Resize to a fixed size
#             img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
#             return img_array
#         else:
#             print(f"Failed to download {image_link}: HTTP status {response.status_code}")
#             return None
#     except Exception as e:
#         print(f"Error processing {image_link}: {e}")
#         return None

# def format_prediction(value, entity_name):
#     # (unchanged, keep your existing format_prediction function here)
#     pass

# def process_batch(batch_df, model):
#     batch_images = []
#     valid_indices = []

#     for index, row in batch_df.iterrows():
#         img_array = process_image(row['image_link'])
#         if img_array is not None:
#             batch_images.append(img_array)
#             valid_indices.append(index)

#     if not batch_images:
#         return []

#     batch_predictions = model.predict(np.array(batch_images))

#     results = []
#     for i, index in enumerate(valid_indices):
#         entity_name = batch_df.loc[index, 'entity_name']
#         prediction = batch_predictions[i][0]
#         formatted_prediction = format_prediction(prediction, entity_name)
#         results.append({
#             'index': index,
#             'prediction': formatted_prediction
#         })

#     return results

# # Load the trained model
# model = load_model("C:/Users/OM/Desktop/Amazon ML School/cnn_regression_model.h5")

# # Load the test data
# test_df = pd.read_csv("C:/Users/OM/Desktop/Amazon ML School/dataset_ml_model/student_resource 3/dataset/test.csv")

# # Make predictions in batches
# batch_size = 32
# all_predictions = []

# for start_idx in tqdm(range(0, len(test_df), batch_size), desc="Processing batches"):
#     end_idx = min(start_idx + batch_size, len(test_df))
#     batch_df = test_df.iloc[start_idx:end_idx]
#     batch_predictions = process_batch(batch_df, model)
#     all_predictions.extend(batch_predictions)

# # Create the output DataFrame
# output_df = pd.DataFrame(all_predictions)

# # Save the output to CSV
# output_df.to_csv('test_out.csv', index=False)

# print("Predictions saved to test_out.csv")

# # Run sanity check
# import subprocess
# subprocess.run(["python", "src/sanity.py", "test_out.csv"])


# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import requests
# from io import BytesIO
# from PIL import Image
# from tqdm.auto import tqdm
# import concurrent.futures
# import gc
# import os

# # Define the allowed units for each entity type (unchanged)
# entity_unit_map = {
#     'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
#     'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
#     'voltage': {'kilovolt', 'millivolt', 'volt'},
#     'wattage': {'kilowatt', 'watt'},
#     'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
# }

# def process_image(image_link):
#     try:
#         response = requests.get(image_link, stream=True, timeout=5)
#         if response.status_code == 200:
#             img = Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224))
#             return np.array(img, dtype=np.float32) / 255.0
#         else:
#             return None
#     except Exception:
#         return None
    

# def format_prediction(value, entity_name):
#     if value <= 0:
#         return ""
    
#     allowed_units = list(entity_unit_map.get(entity_name, set()))
#     if not allowed_units:
#         return ""
    
#     if entity_name in ['width', 'depth', 'height']:
#         if value < 10:
#             unit = 'millimetre'
#         elif value < 1000:
#             unit = 'centimetre'
#             value /= 10
#         else:
#             unit = 'metre'
#             value /= 1000
#     elif entity_name in ['item_weight', 'maximum_weight_recommendation']:
#         if value < 1000:
#             unit = 'gram'
#         elif value < 1000000:
#             unit = 'kilogram'
#             value /= 1000
#         else:
#             unit = 'ton'
#             value /= 1000000
#     elif entity_name == 'voltage':
#         if value >= 1000:
#             unit = 'kilovolt'
#             value /= 1000
#         elif value < 1:
#             unit = 'millivolt'
#             value *= 1000
#         else:
#             unit = 'volt'
#     elif entity_name == 'wattage':
#         if value >= 1000:
#             unit = 'kilowatt'
#             value /= 1000
#         else:
#             unit = 'watt'
#     elif entity_name == 'item_volume':
#         if value < 1000:
#             unit = 'millilitre'
#         elif value < 1000000:
#             unit = 'litre'
#             value /= 1000
#         else:
#             unit = 'gallon'
#             value /= 3785.41  # Convert mL to gallons
#     else:
#         unit = allowed_units[0]  # Default to first unit if entity type is unknown
    
#     return f"{value:.2f} {unit}"



# def process_chunk(chunk_df, model):
#     results = []
#     batch_size = 64
#     for start in range(0, len(chunk_df), batch_size):
#         end = min(start + batch_size, len(chunk_df))
#         batch = chunk_df.iloc[start:end]
        
#         images = []
#         valid_indices = []
#         for index, row in batch.iterrows():
#             img = process_image(row['image_link'])
#             if img is not None:
#                 images.append(img)
#                 valid_indices.append(index)
        
#         if images:
#             batch_predictions = model.predict(np.array(images), verbose=0)
#             for i, index in enumerate(valid_indices):
#                 entity_name = batch.loc[index, 'entity_name']
#                 prediction = batch_predictions[i][0]
#                 formatted_prediction = format_prediction(prediction, entity_name)
#                 results.append({
#                     'index': index,
#                     'prediction': formatted_prediction
#                 })
    
#     return results

# # Load the trained model
# model = load_model("C:/Users/OM/Desktop/Amazon ML School/cnn_regression_model.h5")

# # Load the test data
# test_df = pd.read_csv("C:/Users/OM/Desktop/Amazon ML School/dataset_ml_model/student_resource 3/dataset/test.csv")

# # Process data in chunks
# chunk_size = 1000  # Adjust based on your memory constraints
# num_workers = 4  # Adjust based on your system's capabilities

# all_predictions = []
# output_file = 'test_out.csv'

# # If the output file exists, load existing predictions
# if os.path.exists(output_file):
#     existing_predictions = pd.read_csv(output_file)
#     all_predictions = existing_predictions.to_dict('records')
#     start_index = existing_predictions['index'].max() + 1
# else:
#     start_index = 0

# for chunk_start in tqdm(range(start_index, len(test_df), chunk_size), desc="Processing chunks"):
#     chunk_end = min(chunk_start + chunk_size, len(test_df))
#     chunk_df = test_df.iloc[chunk_start:chunk_end]
    
#     with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
#         future_to_chunk = {executor.submit(process_chunk, chunk_df, model): chunk_df}
        
#         for future in concurrent.futures.as_completed(future_to_chunk):
#             chunk_predictions = future.result()
#             all_predictions.extend(chunk_predictions)
    
#     # Save intermediate results
#     output_df = pd.DataFrame(all_predictions)
#     output_df.to_csv(output_file, index=False)
    
#     # Clear memory
#     gc.collect()

# print(f"All predictions saved to {output_file}")

# # Run sanity check
# import subprocess
# subprocess.run(["python", "src/sanity.py", output_file])





import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
from PIL import Image
from tqdm.auto import tqdm
import concurrent.futures
import gc
import os

# Define the allowed units for each entity type (unchanged)
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

def process_image(image_link):
    try:
        response = requests.get(image_link, stream=True, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Check if the image is blank or unclear
            if np.mean(img_array) > 0.99 or np.mean(img_array) < 0.01:
                return None
            
            return img_array
        else:
            return None
    except Exception:
        return None

def format_prediction(value, entity_name):
    if value <= 0:
        return ""
    
    allowed_units = list(entity_unit_map.get(entity_name, set()))
    if not allowed_units:
        return ""
    if entity_name in ['width', 'depth', 'height']:
        if value < 10:
            unit = 'millimetre'
        elif value < 1000:
            unit = 'centimetre'
            value /= 10
        else:
            unit = 'metre'
            value /= 1000
    elif entity_name in ['item_weight', 'maximum_weight_recommendation']:
        if value < 1000:
            unit = 'gram'
        elif value < 1000000:
            unit = 'kilogram'
            value /= 1000
        else:
            unit = 'ton'
            value /= 1000000
    elif entity_name == 'voltage':
        if value >= 1000:
            unit = 'kilovolt'
            value /= 1000
        elif value < 1:
            unit = 'millivolt'
            value *= 1000
        else:
            unit = 'volt'
    elif entity_name == 'wattage':
        if value >= 1000:
            unit = 'kilowatt'
            value /= 1000
        else:
            unit = 'watt'
    elif entity_name == 'item_volume':
        if value < 1000:
            unit = 'millilitre'
        elif value < 1000000:
            unit = 'litre'
            value /= 1000
        else:
            unit = 'gallon'
            value /= 3785.41  # Convert mL to gallons
    else:
        unit = allowed_units[0]  # Default to first unit if entity type is unknown
    
    return f"{value:.2f} {unit}"
    

def process_chunk(chunk_df, model):
    results = []
    batch_size = 64
    for start in range(0, len(chunk_df), batch_size):
        end = min(start + batch_size, len(chunk_df))
        batch = chunk_df.iloc[start:end]
        
        images = []
        valid_indices = []
        for index, row in batch.iterrows():
            img = process_image(row['image_link'])
            if img is not None:
                images.append(img)
                valid_indices.append(index)
            else:
                # Handle unclear or blank images
                results.append({
                    'index': index,
                    'prediction': " "  # Return a space for unclear images
                })
        
        if images:
            batch_predictions = model.predict(np.array(images), verbose=0)
            for i, index in enumerate(valid_indices):
                entity_name = batch.loc[index, 'entity_name']
                prediction = batch_predictions[i][0]
                formatted_prediction = format_prediction(prediction, entity_name)
                results.append({
                    'index': index,
                    'prediction': formatted_prediction
                })
    
    return results

# Load the trained model
model = load_model("C:/Users/OM/Desktop/Amazon ML School/cnn_regression_model.h5")

# Load the test data
test_df = pd.read_csv("C:/Users/OM/Desktop/Amazon ML School/dataset_ml_model/student_resource 3/dataset/test.csv")

# Process data in chunks
chunk_size = 1000  # Adjust based on your memory constraints
num_workers = 4  # Adjust based on your system's capabilities

all_predictions = []
output_file = 'test_out.csv'

# If the output file exists, load existing predictions
if os.path.exists(output_file):
    existing_predictions = pd.read_csv(output_file)
    all_predictions = existing_predictions.to_dict('records')
    start_index = existing_predictions['index'].max() + 1
else:
    start_index = 0

for chunk_start in tqdm(range(start_index, len(test_df), chunk_size), desc="Processing chunks"):
    chunk_end = min(chunk_start + chunk_size, len(test_df))
    chunk_df = test_df.iloc[chunk_start:chunk_end]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {executor.submit(process_chunk, chunk_df, model): chunk_df}
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_predictions = future.result()
            all_predictions.extend(chunk_predictions)
    
    # Save intermediate results
    output_df = pd.DataFrame(all_predictions)
    output_df.to_csv(output_file, index=False)
    
    # Clear memory
    gc.collect()

print(f"All predictions saved to {output_file}")

# Run sanity check
import subprocess
subprocess.run(["python", "src/sanity.py", output_file])
