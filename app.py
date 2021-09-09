import streamlit as st 
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

# upload the image
st.title('Upload image example')

def load_model():
    model = tf.keras.models.load_model('PATH_FOR_MY_MODEL')
    return model 
    
model = load_model()
st.write('model has been loaded')

uploaded_image = st.file_uploader("Choose an image", type = "jpg")

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded image.', use_column_width = True)
    st.write("")
    st.write("Classifying...")
    label = predict(uploaded_image)
    
#def predict(image_path):
    