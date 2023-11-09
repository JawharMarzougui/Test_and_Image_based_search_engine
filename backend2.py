from flask import Flask, render_template, request
from elasticsearch import Elasticsearch
from PIL import Image
from io import BytesIO
import math
import spacy
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from elasticsearch import Elasticsearch
from googletrans import Translator
import requests



app = Flask(__name__, static_folder='static')

# Create an Elasticsearch client
es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])
index_name = 'flickrphotos'

def translate_text(text, target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

def get_image_embeddings(input_image):
    input_image = input_image.convert('RGB')
    input_image = input_image.resize((224, 224))
    input_image = image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = preprocess_input(input_image)
    image_embedding = vgg16.predict(input_image)
    return image_embedding

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/results', methods=['GET'])
def results():
    query = request.args.get('query')
    
    # Process the query using spaCy to get meaningful keywords and phrases
    doc = nlp(query)
    tags = [token.text for token in doc]
    
    # Extract the page parameter
    page = int(request.args.get('page', 1))
    results_per_page = 50  # Define the number of results per page

    # Calculate the starting index for pagination
    start_index = (page - 1) * results_per_page

    # Construct the Elasticsearch query
    search_body = {
        "from": start_index,
        "size": results_per_page,
        "query": {
            "bool": {
                "must": [
                    {"term": {"tags": tag}} for tag in tags
                ]
            }
        }
    }
    index_name = 'flickrphotos'

    try:
        response = es.search(index=index_name, body=search_body)
        hits = response['hits']['hits']
        return render_template('results2.html', hits=hits, query=query, page=page)
    except Exception as e:
        return f"Error: {e}"
    
@app.route('/search', methods=['POST'])
def search_similar_images():
    search_type = request.form.get('search_type')
    
    if search_type == 'text':
        query = request.form.get('query')
        
        # Translate the query to English
        translated_query = translate_text(query, target_language='en')
        
        doc = nlp(translated_query)
        tags = [token.text for token in doc if not token.is_stop]

        # Extract the page parameter
        page = int(request.form.get('page', 1))
        results_per_page = 50  # Define the number of results per page

        # Calculate the starting index for pagination
        start_index = (page - 1) * results_per_page

        # Construct the Elasticsearch query
        search_body = {
            "from": start_index,
            "size": results_per_page,
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tags": tag}} for tag in tags
                    ]
                }
            }
        }

        try:
            response = es.search(index=index_name, body=search_body)
            hits = response['hits']['hits']
            return render_template('results2.html', hits=hits, query=query, page=page)
        except Exception as e:
            return f"Error: {e}"
    elif search_type == 'image':
        uploaded_file = request.files['image']
        image_url = request.form.get('image_url')

        if uploaded_file.filename != '' or image_url:
            if uploaded_file.filename != '':
                image_data = uploaded_file.read()
                input_image = Image.open(BytesIO(image_data))
            else:
                # If image URL is provided, download the image
                response = requests.get(image_url)
                input_image = Image.open(BytesIO(response.content))

            vector = get_image_embeddings(input_image)

            # Convert numpy array to list
            vector_list = vector[0].tolist()

            # Search for similar images using elastiknn
            query = {
                "query": {
                    "elastiknn_nearest_neighbors": {
                        "vec": vector_list,
                        "field": "vector",
                        "similarity": "angular",
                        "model": "lsh",
                        "candidates": 100
                    }
                }
            }

            results = es.search(index=index_name, body=query, size=20)
            similar_images = results['hits']['hits']

            return render_template('results2.html', similar_images=similar_images)
        else:
            return "No file uploaded or image URL provided."
    else:
        return "Invalid search type."

if __name__ == '__main__':
    app.run(debug=True)
