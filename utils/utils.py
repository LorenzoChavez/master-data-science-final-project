import re
import cv2
from tqdm import tqdm
import os
import shutil
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from scipy.spatial.distance import cdist


def read_image(img_path):
    """ Reads images from a path into an array.
    Fixes exception with B&W images having only 2 colour channels ie. 4540.jpg from street2shop
    Fixes exception with CMYK images having 4 colour channels ie. 4622.jpg from street2shop """
    
    img_array = mpimg.imread(img_path)
    if (len(img_array.shape) == 2) or (img_array.shape[2] > 3):
            img_array = np.array(Image.fromarray(img_array).convert("RGB"))
    return img_array

def bbox_corners(img_path, df):
    """ Reads an img path, captures the photo ID (HAS TO BE UNIQUE IN THE DF) and returns the
    bounding box corners """
    
    img_id = re.split("(\\d+)", img_path)[-2]
    img_df = df[df["photo"] == int(img_id)] #assumes there are unique photo names in the df.photo
    top, left, width, height = (int(img_df["top"]), int(img_df["left"]), int(img_df["width"]), int(img_df["height"]))
    return top, left, width, height

def crop_image(img_path, customer_df, resize):
    """ Takes an image path and returns its image array representation already cropped based on the bbox coordinates from the customer_df.
    Also resizes image to specified value, ie (224, 244) """
    
    top, left, width, height = bbox_corners(img_path, customer_df)
    img_array = read_image(img_path)
    image_cropped = img_array[top:top+height , left:left+width, :] #crops image based on bounding box coordinates
    image_cropped = cv2.resize(image_cropped, dsize=resize, interpolation=cv2.INTER_CUBIC)
    return image_cropped

def remove_gitkeep(path):
    img_paths = os.listdir(path)

    gitFile = ".gitkeep"
    if gitFile in img_paths:  #remove git file so there are only images in the list
        os.remove(path+gitFile)
        img_paths.remove(gitFile)

def find_paths(dataset_path):
    """ Given a path to a set of images from the dataset it returns a list with a full path to each of the images"""
    remove_gitkeep(dataset_path)
    img_paths = os.listdir(dataset_path)
    img_paths = sorted(img_paths, key = lambda x: int(re.sub("(\\D)", "", x))) #sorts numerically rather than alphabetically
    img_paths = [os.path.join(dataset_path, img) for img in img_paths]
    return img_paths

def save_embeddings(dataset_path, file_name, model, preprocessing, shape, resize):
    """Given a path to a set of images it will extract a feature representation for each of the images (embeddings) and will save them all to a file on disk.
    Need to specify the name of the output file, a valid model (List of models: https://keras.io/applications/),
    its preprocessing function, the shape of the feature extraction output of an image (ie. 1024 for MobileNet or 512 for VGG16)
    and the resize factor (ie. (224, 244) for MobileNet)"""
    
    imgs = find_paths(dataset_path)
    embeddings_np = np.zeros((len(imgs), shape)) #creates a matrix of zeros to hold the embeddings
    
    for n, img in enumerate(tqdm(imgs)):
        img_array = read_image(img)
        img_array = cv2.resize(img_array, dsize=resize, interpolation=cv2.INTER_CUBIC)
        img_array = preprocessing(img_array) #preprocess to NN format
        img_array = model.predict(np.expand_dims(img_array, axis=0))  #extracts embedding
        embeddings_np[n] = img_array #appends to previous matrix
    embeddings_np = embeddings_np.astype(np.float16)
    np.save(file_name, embeddings_np) #npy files save/load faster than csv when working with ndarrays
    
def rank_recommendations(img_path, embs_retrieval, model, preprocessing, resize, customer_df=None, bbox=True, method="cosine"):
    """For a single image it extracts its features to an array and computes its distance to all of the arrays in the embeddings file.
    Returns an array of index positions that can be used to sort the images in the embeddings file based on similarity.
    Model and preprocessing must be the same as when the embeddings file was created.
    Using cosine distance as default.
    bbox = True by default, allows to crop images from the original dataset, if False we can use it with any image"""
    
    if bbox == True:
        img_array = crop_image(img_path, customer_df, resize=resize) #crops to bbox and resizes
    else:
        img_array = read_image(img_path)
        img_array = cv2.resize(img_array, dsize=resize, interpolation=cv2.INTER_CUBIC)
    proc_img = preprocessing(np.expand_dims(img_array, axis=0)) #preprocess to NN format
    embs_target = model.predict(proc_img) #extracts embedding

    distance = cdist(embs_retrieval, embs_target.reshape(1,-1), method) #distances from user photo to embeddings file
    rank = np.argsort(distance.ravel()) #ranks by similarity, returns the indices that would sort the distance array
    return rank
    
def load_embeddings(full_path_file):
    """Loads embeddings file"""
    return np.load(full_path_file)
    
def recommend_user(store_database, rank, top_n):   
    """ For use in flask application. Takes the rank from the rank_recommendations function and uses it to order the store_database file based on similarity.
    top_n controls the number of recommendations, maximum of 12 supported by the flask application.
    After ranking the store_database file it will clear duplicate "id" fields so when a product have "n" photos only the first photo is taken into account. Then 
    the "top_n" recommendations are moved to a folder from where the flask app will retrieve the images for display.
    """     
    ranked_database = store_database.iloc[rank] #orders database by similarity
    ranked_database = ranked_database.drop_duplicates(subset=["id"], keep="first") #removes duplicate ids keeping only the most similar photo for each product
    ranked_list = ranked_database["photo"].values.tolist()[:top_n] #takes photo name and indexes top_n recommendations
    
    for n, photo_name in enumerate(ranked_list):
        store_path = os.path.join(".", "static", "images", "store", str(photo_name)+".jpg")
        recommend_path = os.path.join(".", "static", "images", "recommend", str(n)+".jpg") #images are called 0-top_n depending on similarity for the flask template to load them up
        shutil.copyfile(store_path, recommend_path)