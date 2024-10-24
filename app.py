from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = Sequential([model, GlobalMaxPooling2D()])
#model.summary()

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expand_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expand_img)
    result_to_resnet = model.predict(preprocessed_img)
    flatten_result = result_to_resnet.flatten()
    # normalizing
    result_normlized = flatten_result / norm(flatten_result)

    return result_normlized
#print(os.listdir('images'))
filenames = []

for fashion_images in os.listdir('images'):
    images_path = os.path.join('images', fashion_images)
    filenames.append(images_path)

# extracting image features
image_features = []

for files in tqdm(filenames):
    features_list = extract_features(files, model)
    image_features.append(features_list)

pickle.dump(image_features, open('features.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))