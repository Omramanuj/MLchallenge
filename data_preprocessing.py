import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the Dataset
data_path = '/kaggle/input/dataset-ml/train.csv'
data = pd.read_csv(data_path)

print("Dataset Info:")
print(data.info())
print(data.head())

def process_image(image_link):
    try:
        response = requests.get(image_link, stream=True, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure RGB
            img = img.resize((224, 224), Image.LANCZOS)  # Use LANCZOS for high-quality downsampling
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            if img_array.shape != (224, 224, 3):
                print(f"Unexpected shape for {image_link}: {img_array.shape}")
                return None
            return img_array
        else:
            print(f"Failed to download {image_link}: HTTP status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error processing {image_link}: {e}")
        return None

def process_images_in_batches(image_links, labels, entity_names, batch_size=100, max_batches=100):
    all_images = []
    all_labels = []
    all_entity_names = []
    for i in range(0, min(len(image_links), max_batches * batch_size), batch_size):
        batch_links = image_links[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        batch_entity_names = entity_names[i:i+batch_size]
        batch_images = []
        for link, label, entity_name in tqdm(zip(batch_links, batch_labels, batch_entity_names), desc=f"Processing batch {i//batch_size + 1}", total=len(batch_links)):
            img_array = process_image(link)
            if img_array is not None:
                batch_images.append(img_array)
                all_labels.append(label)
                all_entity_names.append(entity_name)
        if batch_images:
            all_images.extend(batch_images)
            print(f"Batch {i//batch_size + 1}:")
            print(f"  Number of images: {len(batch_images)}")
            if len(batch_images) > 0:
                print(f"  Shape: {np.array(batch_images[0]).shape}")
                print(f"  Min value: {np.min(batch_images):.4f}")
                print(f"  Max value: {np.max(batch_images):.4f}")
                print(f"  Mean value: {np.mean(batch_images):.4f}")
                print(f"  Standard deviation: {np.std(batch_images):.4f}")
            print()
        
        if i//batch_size + 1 >= max_batches:
            break
    
    return np.array(all_images), np.array(all_labels), np.array(all_entity_names)

# Process images in the training set
image_links = data['image_link'].values
labels = data['entity_value'].values
entity_names = data['entity_name'].values

# Use LabelEncoder for labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, y_train, entity_names_train = process_images_in_batches(image_links, encoded_labels, entity_names, max_batches=50)

# Save the processed data
np.save('X_train_50batches.npy', X_train)
np.save('y_train_50batches.npy', y_train)
np.save('entity_names_50batches.npy', entity_names_train)

# Save the label encoder
joblib.dump(label_encoder, 'label_encoder_50.joblib')

print("Processed data saved as 'X_train_50batches.npy', 'y_train_50batches.npy', and 'entity_names_50batches.npy'")
print("Label encoder saved as 'label_encoder_50.joblib'")
