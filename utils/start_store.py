import os
import shutil
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from keras.applications.mobilenet import MobileNet, preprocess_input


def start_store(seed, csvs_path, dataset_path, store_path):
    np.random.seed(seed)

    customer_df = pd.read_csv(os.path.join(csvs_path, "customer_df.csv"))
    retrieval_df = pd.read_csv(os.path.join(csvs_path, "retrieval_df.csv"))
    
    #filtering by unique photos
    freq_categories = pd.DataFrame(customer_df["photo"].value_counts())
    freq_categories = freq_categories[freq_categories["photo"] == 1]
    customer_df = customer_df[customer_df["photo"].isin(freq_categories.index)]
    
    #merging both datasets to get rid of the retrieval products that do not match
    matches_df = customer_df.merge(retrieval_df, how="inner", on="id", suffixes=("_cust", "_retr"))
    matches_df = matches_df[matches_df["category_cust"].isin(["dresses", "tops"])]
    
    #takes half the set, fixed by seed
    perm = np.random.permutation(matches_df['product_cust'].unique()) #permutes list of photos randomly
    split = int(len(perm)*0.5)
    list_products = perm[:split]
    products_df = matches_df[matches_df['product_cust'].isin(list_products)]
    products_df_final = retrieval_df[retrieval_df['photo'].isin(products_df["photo_retr"].unique())]
    
    photos_list = products_df_final["photo"].tolist() #photo names are already unique
    photos_list = [str(photo) + ".jpg" for photo in photos_list]
    print(photos_list)
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)    
    
    for photo_name in tqdm(photos_list):
        shutil.copy(os.path.join(dataset_path, photo_name), os.path.join(store_path, photo_name))
                
    products_df_final = products_df_final.sort_values(by="photo")
    products_df_final.to_csv(os.path.join("..", "utils", "products.csv"), index=False)

def find_paths(dataset_path):
    img_paths = os.listdir(dataset_path)
    img_paths = sorted(img_paths, key = lambda x: int(re.sub("(\\D)", "", x))) #sorts numeric rather than str
    img_paths = [os.path.join(dataset_path, img) for img in img_paths]
    return img_paths
    
def save_embeddings(dataset_path, resize=(224,224)):
    img_paths = find_paths(dataset_path)    
    model = MobileNet(weights="imagenet", include_top=False, pooling="avg")
    embeddings_tensor = np.zeros((len(img_paths), 1024))
    
    with tqdm(total=len(img_paths)) as pbar:
        for n, img in enumerate(img_paths):
            img_array = plt.imread(img)
            img_array = cv2.resize(img_array, dsize=resize, interpolation=cv2.INTER_CUBIC)
            img_array = preprocess_input(img_array)
            img_array = model.predict(np.expand_dims(img_array, axis=0)) #default ouput shape mobilenet (6,6,1024)
            embeddings_tensor[n] = img_array.flatten()
            pbar.update(1)
    embeddings_tensor = embeddings_tensor.astype(np.float16)
    
    np.save("../embeddings.npy", embeddings_tensor) #npy files save/load faster than csv when working with ndarrays
    
if __name__ == "__main__":
    seed = 2018
    csvs_path = os.path.join("..", "..")
    dataset_path = os.path.join(".." , "..", "photos_resized")
    store_path = os.path.join("..", "static", "images", "store")
    start_store(seed, csvs_path, dataset_path, store_path)
    save_embeddings(store_path)