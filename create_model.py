import tensorflow as tf
import numpy as np
import os

# Define the class names
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

print(f"Creating model for {len(class_names)} classes")

# Create a new model with the same architecture as the training notebook
model = tf.keras.Sequential([
    # First Conv Block
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
    # Second Conv Block
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
    # Third Conv Block
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
    # Fourth Conv Block
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
    # Fifth Conv Block
    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    
    # Dropout and Dense layers
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1500, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(units=len(class_names), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save the model
model.save('new_plant_disease_model.keras', save_format='keras')
print("Model saved as 'new_plant_disease_model.keras'")

# Print model summary
model.summary() 