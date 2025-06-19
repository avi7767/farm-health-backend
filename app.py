from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import json
import requests
from PIL import Image
import io
import base64
import pickle
import random
from datetime import datetime
from dotenv import load_dotenv
import logging
import sys
import traceback
from werkzeug.utils import secure_filename

# Configure TensorFlow for better performance
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.set_soft_device_placement(True)

# Enable mixed precision for faster computation
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flask_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("Starting application...")

# Initialize Flask app
app = Flask(__name__)

# Allow your Netlify frontend
CORS(app, resources={r"/*": {"origins": "https://farmhealth.netlify.app"}})

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Flask app and CORS initialized")

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Get API keys from environment variables
nvidia_api_key = os.getenv('NVIDIA_API_KEY')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# Log API key status (masked for security)
if nvidia_api_key:
    masked_key = nvidia_api_key[:8] + '...' + nvidia_api_key[-4:]
    logger.info(f"NVIDIA API key loaded: {masked_key}")
    print(f"NVIDIA API key loaded: {masked_key}")
else:
    logger.error("NVIDIA API key not found in environment variables")
    print("NVIDIA API key not found in environment variables")

if not WEATHER_API_KEY:
    print("Warning: WEATHER_API_KEY not found in environment variables")
else:
    print("WEATHER_API_KEY loaded successfully")

# Load class names
class_names = [
    'Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
    'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew',
    'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy',
    'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
    'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
    'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
    'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
    'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
    'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus',
    'Tomato_healthy'
]
print(f"Class names loaded: {len(class_names)} classes")

# Function to preprocess image
def preprocess_image(image):
    try:
        # Log input image details
        logger.info(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            logger.info("Converting grayscale to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            logger.info("Converting RGBA to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # Check if it's BGR (OpenCV default) and convert to RGB
            logger.info("Converting BGR to RGB")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image to 224x224 (matching model's expected input size)
        logger.info("Resizing image to 224x224")
        image = cv2.resize(image, (224, 224))
        
        # Normalize pixel values
        logger.info("Normalizing pixel values")
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        logger.info("Adding batch dimension")
        image = np.expand_dims(image, axis=0)
        
        # Log final image details
        logger.info(f"Final image shape: {image.shape}, dtype: {image.dtype}, range: [{np.min(image):.3f}, {np.max(image):.3f}]")
        
        return image
        
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

# Function to get weather data
def get_weather_data(city):
    try:
        logger.info(f"Fetching weather data for {city}...")
        response = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        
        logger.info(f"Weather API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": round(data["main"]["temp"], 1),
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "description": data["weather"][0]["description"].capitalize()
            }
        else:
            error_msg = f"Weather API Error: Status code {response.status_code}"
            logger.error(error_msg)
            logger.error(f"Response text: {response.text}")
            return None
    except Exception as e:
        error_msg = f"Weather API Error: {str(e)}"
        logger.error(error_msg)
        return None

# Function to get chatbot response
def get_chatbot_response(prompt, selected_language):
    try:
        # Log the API key status (masked for security)
        if nvidia_api_key:
            masked_key = nvidia_api_key[:8] + '...' + nvidia_api_key[-4:]
            logger.info(f"Using NVIDIA API key: {masked_key}")
        else:
            logger.error("NVIDIA API key is missing!")
            return "Error: API key not configured"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {nvidia_api_key}"
        }
        
        # Add language instruction to the prompt
        language_prompt = f"""You must respond ONLY in {selected_language} language. Never use any other language in your response.

{prompt}"""
        
        data = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert plant pathologist and agricultural advisor. You must respond ONLY in {selected_language} language. Never use any other language in your response."
                },
                {
                    "role": "user",
                    "content": language_prompt
                }
            ],
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 1024,
            "stream": False
        }
        
        logger.info(f"Making AI API request with prompt: {prompt[:100]}...")
        logger.info(f"Request URL: https://integrate.api.nvidia.com/v1/chat/completions")
        logger.info(f"Request headers: {headers}")
        logger.info(f"Request data: {data}")
        
        # Use the correct API endpoint
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        logger.info(f"AI API Response Status: {response.status_code}")
        logger.info(f"AI API Response Headers: {response.headers}")
        logger.info(f"AI API Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            advice = result["choices"][0]["message"]["content"].strip()
            logger.info(f"Successfully received advice: {advice[:100]}...")
            return advice
        else:
            error_msg = f"AI API Error: Status code {response.status_code}"
            logger.error(error_msg)
            logger.error(f"Response text: {response.text}")
            return f"Error getting advice: {error_msg}"

    except requests.exceptions.Timeout:
        error_msg = "AI API request timed out after 30 seconds"
        logger.error(error_msg)
        return f"Error getting advice: {error_msg}"
    except requests.exceptions.RequestException as e:
        error_msg = f"AI API request failed: {str(e)}"
        logger.error(error_msg)
        return f"Error getting advice: {error_msg}"
    except Exception as e:
        error_msg = f"AI API Error: {str(e)}"
        logger.error(error_msg)
        return f"Error getting advice: {error_msg}"

# Load the model at startup
def load_model():
    try:
        # Get the absolute path to the model file
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plant_disease_model.h5')
        logger.info(f"Attempting to load model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        # Load the model
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

# Function to get treatment advice from AI
def get_treatment_advice(disease_name, language='English'):
    """Get treatment advice for the detected disease using AI API"""
    try:
        # Log the API key status (masked for security)
        if nvidia_api_key:
            masked_key = nvidia_api_key[:8] + '...' + nvidia_api_key[-4:]
            logger.info(f"Using NVIDIA API key: {masked_key}")
        else:
            logger.error("NVIDIA API key is missing!")
            return "Error: API key not configured"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {nvidia_api_key}"
        }
        
        prompt = f"""You are an expert plant pathologist. Provide detailed treatment advice for {disease_name} in {language} language only. Do not use any other language.

        Include the following information in {language}:
        1. Immediate actions to take
        2. Recommended treatments/medicines
        3. Preventive measures
        4. Expected recovery time
        5. When to seek professional help
        
        Format the response in clear, easy-to-follow steps. Remember to respond ONLY in {language} language."""
        
        data = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are an expert plant pathologist and agricultural advisor. You must respond ONLY in {language} language. Never use any other language in your response."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 1024,
            "stream": False
        }
        
        logger.info(f"Making AI API request for disease: {disease_name} in {language}")
        logger.info(f"Request URL: https://integrate.api.nvidia.com/v1/chat/completions")
        logger.info(f"Request headers: {headers}")
        logger.info(f"Request data: {data}")
        
        # Use the correct API endpoint
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        logger.info(f"AI API Response Status: {response.status_code}")
        logger.info(f"AI API Response Headers: {response.headers}")
        logger.info(f"AI API Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                advice = result["choices"][0]["message"]["content"].strip()
                logger.info(f"Successfully received treatment advice for {disease_name} in {language}")
                return advice
            else:
                error_msg = "Invalid response format from AI API"
                logger.error(error_msg)
                logger.error(f"Response: {result}")
                return f"Error getting treatment advice: {error_msg}"
        else:
            error_msg = f"AI API Error: Status code {response.status_code}"
            logger.error(error_msg)
            logger.error(f"Response text: {response.text}")
            return f"Error getting treatment advice: {error_msg}"

    except requests.exceptions.Timeout:
        error_msg = "AI API request timed out after 30 seconds"
        logger.error(error_msg)
        return f"Error getting treatment advice: {error_msg}"
    except requests.exceptions.RequestException as e:
        error_msg = f"AI API request failed: {str(e)}"
        logger.error(error_msg)
        return f"Error getting treatment advice: {error_msg}"
    except Exception as e:
        error_msg = f"Error getting treatment advice: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Function to predict disease
def predict_disease(image_path):
    try:
        # Load the model
        model = load_model()
        if model is None:
            raise ValueError("Failed to load model")
        
        # Read and preprocess the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to read image")
            
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get disease name
        disease_name = class_names[predicted_class]
        
        return {
            'disease': disease_name,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Error in predict_disease: {str(e)}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise

# Route for disease prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    """Handle disease prediction requests"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get the selected language from the request
        selected_language = request.form.get('language', 'English')
        
        # Save the uploaded file temporarily
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(temp_path)
        
        try:
            # Get prediction from model
            prediction = predict_disease(temp_path)
            
            if prediction:
                disease_name = prediction['disease']
                confidence = prediction['confidence']
                
                # Get treatment advice in the selected language
                treatment = get_treatment_advice(disease_name, selected_language)
                
                return jsonify({
                    'disease': disease_name,
                    'confidence': confidence,
                    'treatment': treatment
                })
            else:
                return jsonify({'error': 'Failed to get prediction'}), 500
                
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route for expert advice
@app.route('/api/expert-advice', methods=['POST', 'OPTIONS'])
def get_expert_advice():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        print("Received request data:", data)
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        city = data.get('city')
        crop_type = data.get('cropType')
        soil_type = data.get('soilType')
        language = data.get('language', 'English')
        
        if not all([city, crop_type, soil_type]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        print(f"Getting advice for {crop_type} in {city} with {soil_type} soil")
        
        # Get weather data
        weather_data = get_weather_data(city)
        if not weather_data:
            return jsonify({'error': 'Failed to fetch weather data'}), 500
        
        # Get AI advice
        prompt = f"""
        You are an expert agricultural advisor. Provide detailed farming advice for {crop_type} in {language} language only. Do not use any other language.

        Consider the following details:
        - Location: {city}
        - Soil type: {soil_type}
        - Current weather: {weather_data['description']}
        - Temperature: {weather_data['temp']}°C
        - Humidity: {weather_data['humidity']}%
        - Wind speed: {weather_data['wind_speed']} m/s
        
        Include the following sections in {language}:
        1. Best planting practices
        2. Soil preparation tips
        3. Water management
        4. Fertilizer recommendations
        5. Pest control measures
        6. Weather-specific precautions
        7. Expected yield and harvest time

        Remember to provide the entire response in {language} language only.
        """
        
        advice = get_chatbot_response(prompt, language)
        if not advice:
            return jsonify({'error': 'Failed to get AI advice'}), 500
        
        return jsonify({
            'advice': advice,
            'weather': weather_data
        })
    
    except Exception as e:
        print(f"Error in expert advice: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Route for testing API connection
@app.route('/api/test-ai', methods=['GET', 'OPTIONS'])
def test_ai():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Check if API key is loaded
        if not nvidia_api_key:
            error_msg = "NVIDIA API key not found in environment variables"
            logger.error(error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
        
        # Log the API key status (masked for security)
        masked_key = nvidia_api_key[:8] + '...' + nvidia_api_key[-4:]
        logger.info(f"Testing AI API with key: {masked_key}")

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {nvidia_api_key}'
        }
        
        data = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": "Hello, this is a test message."
                }
            ],
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 100,
            "stream": False
        }
        
        logger.info("Sending test request to NVIDIA API...")
        logger.info(f"Request URL: https://integrate.api.nvidia.com/v1/chat/completions")
        logger.info(f"Request headers: {headers}")
        logger.info(f"Request data: {data}")
        
        # Use the correct API endpoint
        response = requests.post(
            'https://integrate.api.nvidia.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        logger.info(f"NVIDIA API Response Status: {response.status_code}")
        logger.info(f"NVIDIA API Response Headers: {response.headers}")
        logger.info(f"NVIDIA API Response: {response.text}")
        
        if response.status_code == 200:
            logger.info("AI API test successful")
            return jsonify({
                "status": "success",
                "message": "AI API connection successful",
                "response": response.json()
            })
        else:
            error_msg = f"AI API returned error status: {response.status_code}"
            logger.error(error_msg)
            logger.error(f"Response text: {response.text}")
            return jsonify({
                "status": "error",
                "message": error_msg,
                "details": response.text
            }), 500
        
    except requests.exceptions.Timeout:
        error_msg = "AI API request timed out after 30 seconds"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500
    except requests.exceptions.RequestException as e:
        error_msg = f"AI API request failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500
    except Exception as e:
        error_msg = f"AI API connection error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "status": "error",
            "message": error_msg
        }), 500

# Route for testing API connection
@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {nvidia_api_key}"
        }
        
        data = {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello and confirm you are working."}
            ],
            "temperature": 0.5,
            "top_p": 1,
            "max_tokens": 100,
            "stream": False
        }
        
        print("Testing API connection...")
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("API connection successful")
            return jsonify({
                "status": "success",
                "message": "API connection successful",
                "response": result["choices"][0]["message"]["content"].strip()
            })
        else:
            print(f"API Error: Status code {response.status_code}")
            return jsonify({
                "status": "error",
                "message": f"API Error: Status code {response.status_code}",
                "details": response.text
            }), 500
        
    except Exception as e:
        print(f"API connection error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "API connection failed",
            "details": str(e)
        }), 500

# Route for getting translations
@app.route('/api/translations', methods=['GET'])
def get_translations():
    try:
        language = request.args.get('language', 'English')
        print(f"Getting translations for language: {language}")
        
        # Import translations from the translations module
        from translations import translations
        
        # Return translations for the requested language, or English as fallback
        return jsonify(translations.get(language, translations['English']))
    except Exception as e:
        print(f"Error getting translations: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add a test route
@app.route('/test')
def test():
    return jsonify({"status": "ok", "message": "Flask server is running"})

# Add this after your existing imports
UPLOAD_FOLDER = 'Dataset'

@app.route('/api/upload-disease', methods=['POST'])
def upload_disease():
    try:
        if 'diseaseName' not in request.form:
            return jsonify({'error': 'Disease name is required'}), 400

        disease_name = request.form['diseaseName']
        disease_folder = os.path.join(UPLOAD_FOLDER, disease_name)

        # Create disease folder if it doesn't exist
        if not os.path.exists(disease_folder):
            os.makedirs(disease_folder)

        # Get all image files from the request
        uploaded_files = []
        for key in request.files:
            if key.startswith('image'):
                file = request.files[key]
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    # Add timestamp to filename to make it unique
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{timestamp}_{filename}"
                    file_path = os.path.join(disease_folder, filename)
                    file.save(file_path)
                    uploaded_files.append(filename)

        if not uploaded_files:
            return jsonify({'error': 'No images were uploaded'}), 400

        return jsonify({
            'message': 'Images uploaded successfully',
            'uploaded_files': uploaded_files,
            'disease_folder': disease_folder
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 