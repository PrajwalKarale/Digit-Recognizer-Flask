from flask import Flask,render_template,request,url_for
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from werkzeug.utils import secure_filename
import sys
import os
import cv2
from tensorflow import keras
from keras import backend as K
app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "static/image/uploads"
model = keras.models.load_model('./model/digit_recognizer.h5')
print('model loaded')

# global graph
# graph = tf.get_default_graph()
@app.route("/")
def home():
    return render_template('home.html')

@app.route("/prediction",methods=["GET", "POST"])
def prediction():
    if request.method == 'POST':
        print('Request accessed')
        image = request.files["file-ip-1"] 
        print(type(image))
        filename = secure_filename(image.filename)
        #image file path
        image_file = "static/image/uploads/"+str(filename)
        # print(type(filename))
        print(f'Filename is {image_file}')
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
        print('File saved')
        og_image = cv2.imread("static/image/uploads/"+str(filename))
        # print(type(og_image))
        grayImage = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
        blackAndWhiteImage = cv2.resize(blackAndWhiteImage, (28, 28), interpolation=cv2.INTER_AREA)
        print(type(blackAndWhiteImage))
        print(blackAndWhiteImage.shape)
        blackAndWhiteImage = blackAndWhiteImage.reshape((1, 28, 28, 1))
        print(blackAndWhiteImage.shape)
        # K.reset_uids()
        # with graph.as_default():
        prediction = model.predict_classes(blackAndWhiteImage)
        print(type(prediction))
        # print(prediction)
        pred = prediction[0]
        print(pred)
        pred_text = "THE DIGIT IS: " + str(pred)
        return render_template('home.html',image_file = image_file,pred_text = pred_text)
    return render_template('home.html')
if __name__ == '__main__':
    app.run()