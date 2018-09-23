import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from scipy.spatial.distance import cdist


def load_embeddings(full_path_file):
    return np.load(full_path_file)
    
class Recommender:
    def __init__(self):
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        self.store_database = pd.read_csv("./utils/products.csv") #NEED TO CREATE A BETTER PROCESS TO OBTAIN THIS FILE

    def recommend_user(self, user_img, embs_store, imgs_store, top_n, method="cosine"):        
        img_array = plt.imread(user_img)
        img_array = cv2.resize(img_array, dsize=(250,250), interpolation=cv2.INTER_CUBIC)
        
        if len(img_array.shape) == 2: #fixes exception with B&W images having only 2 colour channels ie. 4540.jpg from street2shop
            img_array = np.array(Image.fromarray(img_array).convert("RGB"))
        if img_array.shape[2] > 3: #fixes exception with CMYK images having 4 colour channels ie. 4622.jpg from street2shop
            img_array = np.array(Image.fromarray(img_array).convert("RGB"))
        
        proc_img = preprocess_input(np.expand_dims(img_array, axis=0)) #preprocessing user image
        emb_user = self.model.predict(proc_img) #extracts vector representation (embedding)
        distance = cdist(embs_store, emb_user [0].reshape(1,-1), method) #cosine distance between user image and store images
        rank = np.argsort(distance.ravel()) #ranks by similarity, returns the indices that sort the store database
        rank = rank[:top_n]
        
        imgs_ranked = [imgs_store[index].split(".")[0] for index in rank] #extracts photo names
        print(imgs_ranked)
        
        ranked_database = self.store_database #orders database by similarity position
        ranked_database = ranked_database[ranked_database["photo"].isin(imgs_ranked)]
        ranked_database = ranked_database.drop_duplicates(subset=["id"], keep="first") #NEED TO IMPROVE THIS, DOES NOT TAKE THE MOST SIMILAR PHOTO OF THE PRODUCT JUST THE FIRST ONE
        ranked_list = ranked_database["photo"].tolist()[:top_n]
        
        for n, photo_name in enumerate(ranked_list): #uses rank to select the similar images from the catalogue
            store_path = os.path.join(".", "static", "images", "store", str(photo_name)+".jpg")
            recommend_path = os.path.join(".", "static", "images", "recommend", str(n)+".jpg")
            shutil.copyfile(store_path, recommend_path)
