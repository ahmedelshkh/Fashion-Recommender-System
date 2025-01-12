import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


feature_list = pickle.load(open('features.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('fashion recommender system')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expand_img)
    flatten_result = model.predict(preprocessed_img).flatten()

    normalized_result = flatten_result / norm(flatten_result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])
    return indices


uploaded_file = st.file_uploader('Choose an image')

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # file has been uploaded
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        features = feature_extraction(os.path.join('uploads',uploaded_file.name),model)
        st.text(features)

        indices = recommend(features, feature_list)
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("error")
