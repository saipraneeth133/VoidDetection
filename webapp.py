from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from werkzeug.utils import secure_filename
import os
import pandas as pd
import time

df=pd.DataFrame()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'static/uploads/'
application = Flask(__name__, static_folder = UPLOAD_FOLDER)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
aisles = []
spaces = []

class TestConfig(Config):
    NAME = "void_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1

rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
rcnn.load_weights('mask_rcnn_void_cfg_00052.h5', by_name=True)
rcnn.keras_model._make_predict_function()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_image_with_boxes(filename, boxes_list):
    fig = pyplot.figure(figsize=(15, 15))
    fig.add_subplot(1, 2, 1)
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    fig.add_subplot(1, 2, 2)
    pyplot.imshow(data)
    ax = pyplot.gca()
    for box in boxes_list:
        y1, x1, y2, x2 = box
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        ax.add_patch(rect)
    aisles.append(filename)
    spaces.append(len(boxes_list))
    dict = {'aisle name': aisles, 'void number': spaces}
    df = pd.DataFrame(dict)
    df.to_csv(os.path.join(application.config['UPLOAD_FOLDER'], "aisle_summary.csv"))
    pyplot.show()
    #pyplot.savefig(os.path.join(application.config['UPLOAD_FOLDER'], "void.jpg"))

@application.route('/')
def upload():
    return render_template('modelapp.html')

@application.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print (filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            img = load_img(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            img = img_to_array(img)
            results = rcnn.detect([img], verbose=0)
            draw_image_with_boxes(file, results[0]['rois'])
            return render_template('modelapp.html')
            #return render_template('modelapp.html', filenames=names, select_cat="void")
    return render_template('modelapp.html')

if __name__ == "__main__":
    application.run(debug=True, use_reloader=False)
