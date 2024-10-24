import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors
import cv2


feature_list = pickle.load(open('features.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))


model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('samples/Bottle.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expand_img = np.expand_dims(img_array,axis=0)
preprocessed_img = preprocess_input(expand_img)
flatten_result = model.predict(preprocessed_img).flatten()

normalized_result = flatten_result / norm(flatten_result)

neighbors = NearestNeighbors(n_neighbors= 6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])

for file in indices[0][1:6]:
    extract_img = cv2.imread(filenames[file])
    cv2.imshow('sim_img', cv2.resize(extract_img , (512,512)))
    cv2.waitKey(0)