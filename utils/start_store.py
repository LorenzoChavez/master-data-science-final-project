import os
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.applications.mobilenet import MobileNet, preprocess_input
from utils import save_embeddings, remove_gitkeep


def start_store(seed, csvs_path, dataset_path, store_path):
    """Initiates everything required to make the flask app work, such as: 
        Creating the dataframe of products for the store, moving the product images to the store directory.
    Seed by default is 2018 which matches the validation set, for a different set of photos change the seed.
    """
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
    if not os.path.exists(store_path):
        os.makedirs(store_path, exist_ok=True)    
    
    for photo_name in tqdm(photos_list):
        shutil.copy(os.path.join(dataset_path, photo_name), os.path.join(store_path, photo_name))
                
    products_df_final = products_df_final.sort_values(by="photo")
    products_df_final.to_csv(os.path.join("..", "utils", "products.csv"), index=False)
    
if __name__ == "__main__":
    seed = 2018
    csvs_path = os.path.join("..", "notebooks")
    dataset_path = os.path.join("..", "photos_resized")
    store_path = os.path.join("..", "static", "images", "store")
    
    remove_gitkeep(store_path) #removes gitkeep files
    remove_gitkeep(os.path.join(".", "static", "images", "recommend"))
    remove_gitkeep(os.path.join(".", "static", "images", "user"))

    start_store(seed, csvs_path, dataset_path, store_path) #creating the store dataframe and placing the images on the store directory

    resizing = (224,224)
    shape_output = 1024
    model = MobileNet(input_shape=(224, 224, 3), weights="imagenet", include_top=False, pooling="avg")
    save_embeddings(store_path, "embeddings.npy", model, preprocess_input, shape_output, resizing) #creating the embeddings file for the store imagescess_input, shape_output, resizing) #creating the embeddings file for the store imagescess_input, shape_output, resizing) #creating the embeddings file for the store images