from flask import Flask, render_template, request
from utils.utils import load_embeddings, rank_recommendations, recommend_user
from keras.applications.mobilenet import MobileNet, preprocess_input
import pandas as pd

app = Flask(__name__)

embs_store = load_embeddings('./utils/embeddings.npy') # load embeddings from previously created file
user_img_path = "./static/images/user/user_img.jpg" 
model = MobileNet(weights="imagenet", include_top=False, pooling="avg")
store_database = pd.read_csv("./utils/products.csv")

@app.route("/")
@app.route("/home")
def upload_page():
    return render_template("index.html")

@app.after_request
def fix_cache(response): #https://blog.sneawo.com/blog/2017/12/20/no-cache-headers-in-flask/
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/upload', methods = ['POST'])
def upload_image():
    f = request.files['file']
    f.save(user_img_path) #saves uploaded image

    top_n = 12
    rank = rank_recommendations(user_img_path, embs_store, model, preprocess_input, (224,224), bbox=False)
    recommend_user(store_database, rank, top_n)
    return render_template("gallery.html")
#
if __name__ == "__main__":
    app.run(debug=False, threaded=False)
