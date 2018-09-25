from flask import Flask, render_template, request
from utils.utils import load_embeddings, Recommender

app = Flask(__name__)

embs_store = load_embeddings('./embeddings.npy') # load embeddings from previously created file
user_img = "user_img.jpg"
user_img_path = "./static/images/user/" + user_img
reco = Recommender()

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
    reco.recommend_user(user_img_path, embs_store, top_n) #will save in the recommend folder the top_n most similar images
    return render_template("gallery.html")
#
if __name__ == "__main__":
    app.run(debug=False, threaded=False)
