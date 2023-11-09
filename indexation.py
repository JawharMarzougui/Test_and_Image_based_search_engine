from elasticsearch import Elasticsearch
import csv
import numpy as np 
from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import requests
from io import BytesIO

# Initialize Elasticsearch connection
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])

# Define the index name
index_name = 'flickrphotos'


mapping = {
    "dynamic": False,
    "properties": {
        "id": {"type": "keyword"},
        "url": {"type": "text"},
        "vector": {
            "type": "elastiknn_dense_float_vector",
            "elastiknn": {
                "dims": 512,
                "similarity": "angular",
                "model": "lsh",
                "L": 60,
                "k": 3
            }
        },
        "tags": {"type": "text"}  # Add this field for tags
    }
}


# Apply the mapping
es.indices.put_mapping(index=index_name, body=mapping)

vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

def get_image_embeddings(input_image):
    input_image = input_image.convert('RGB')  # Convert to RGB if not already
    input_image = input_image.resize((224, 224))
    input_image = image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = preprocess_input(input_image)
    image_embedding = vgg16.predict(input_image)
    return image_embedding

# Initialize the checkpoint
checkpoint_file = 'checkpoint.txt'

try:
    with open(checkpoint_file, 'r') as f:
        last_processed_row = int(f.read().strip())
except (ValueError, FileNotFoundError):
    last_processed_row = 0

with open('photo_metadata.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    k = 1
    for i, row in enumerate(reader):
        if i < last_processed_row:
            continue
        
        photo_id = row['id']
        farm = row['flickr_farm']
        server = row['flickr_server']
        secret = row['flickr_secret']
        
        photo_url = f"http://farm{farm}.staticflickr.com/{server}/{photo_id}_{secret}.jpg"
        
        response = requests.get(photo_url)
        
        if response.status_code == 200:
            input_image = Image.open(BytesIO(response.content))
            vector = get_image_embeddings(input_image)

            # Convert numpy array to list
            vector_list = vector[0].tolist()  

            tags = row['tags']

            doc = {
            "id": photo_id,
            "url": photo_url,
            "vector": vector_list,
            "tags": tags  # Include the 'tags' field in the document
            }

            action = {
                "_index": index_name,
                "_source": doc
            }

            es.index(index=index_name, body=doc)
            print(f"images: {k}")
        else:
            print(f"Failed to fetch image from {photo_url}. Status code: {response.status_code}")

        with open(checkpoint_file, 'w') as f:
            f.write(str(i + 1))  # Update the checkpoint

        k += 1
