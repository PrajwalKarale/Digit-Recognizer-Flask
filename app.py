from flask import Flask,render_template,request,url_for
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from werkzeug.utils import secure_filename
import sys
import os
import cv2
from tensorflow import keras
from keras import backend as K
# import matplotlib.pyplot as plt
app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "static/image/uploads"
# app.config["IMAGE_UPLOADS"] = "static/image/bounding"
model = keras.models.load_model('./model/digit_recognizer.h5')
print('model loaded')

# global graph
# graph = tf.get_default_graph()
def bounding_box(bound_image,bounded_image_display,pred,filename):
    #resizing the orignal image to (170,170,3)
    bounded_image_display = cv2.resize(bounded_image_display,(170,170))
    bound_image = cv2.resize(bound_image,(170,170))
    bound_image=cv2.cvtColor(bound_image,cv2.COLOR_BGR2GRAY)
    (thresh, img_bin) = cv2.threshold(bound_image, 128, 255, cv2.THRESH_OTSU)
    print(thresh)
    img_bin = 255-img_bin
    contours0, hierarchy = cv2.findContours(img_bin.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
    countours_largest = sorted(contours, key=lambda x: cv2.contourArea(x))[-1]
    bb=cv2.boundingRect(countours_largest)
    print(bb)
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,255,0)
    org = (20, 20)
    fontScale = 0.5
    thickness = 1
    pt1 = (bb[0],bb[1]) # upper coordinates 
    pt2 = (bb[0]+bb[2],bb[1]+bb[3]) # lower coordinates
    img_gray_bb = bounded_image_display.copy()
    img_gray_bb = cv2.resize(img_gray_bb,(170,170))
    # print(type(img_gray_bb))
    # print(img_gray_bb.shape)
    # print(img_gray_bb)
    text = "DIGIT: "+ str(pred)
    cv2.rectangle(img_gray_bb,pt1,pt2,color,1)
    cv2.putText(img_gray_bb, text, org, font,fontScale, color, thickness, cv2.LINE_AA)
    path = '/home/prajwal/Desktop/flask_pro/digit_recognizer/digit_app/static/image/bounding'
    filename = str(filename)
    cv2.imwrite(os.path.join(path , filename), img_gray_bb)
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
        #storing in temporary variable for the bounding box image
        bound_image = og_image
        bounded_image_display = og_image
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
        bounded_image = bounding_box(bound_image,bounded_image_display,pred,filename)
        img='static/image/bounding/'+str(filename)
        return render_template('home.html',image_file = image_file,pred_text = pred_text,img = img)
    return render_template('home.html')
if __name__ == '__main__':
    app.run()
