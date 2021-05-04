# example of inference with a pre-trained coco model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import pandas as pd

# draw an image with detected objects
def draw_image_with_boxes(filename, boxes_list):
     fig = pyplot.figure(figsize=(10, 7))
     fig.add_subplot(2, 1, 1)
     data = pyplot.imread(filename)
     pyplot.imshow(data)
     fig.add_subplot(2, 1, 2)
     # plot the image
     pyplot.imshow(data)
     # get the context for drawing boxes
     ax = pyplot.gca()
     # plot each box
     for box in boxes_list:
          # get coordinates
          y1, x1, y2, x2 = box
          # calculate width and height of the box
          width, height = x2 - x1, y2 - y1
          # create the shape
          rect = Rectangle((x1, y1), width, height, fill=False, color='red')
          # draw the box
          ax.add_patch(rect)
     # show the plot
     '''
     dict = {'aisle name': aisles, 'void number': spaces}
     df = pd.DataFrame(dict)
     df.to_csv(
          r'C:\\Users\sai\Downloads\train_yolo_to_detect_custom_object\yolo_custom_detection\aisle_summary.csv')
     '''
     pyplot.show()

# define the test configuration
class TestConfig(Config):
     NAME = "void_cfg"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 1

# define the model
rcnn = MaskRCNN(mode='inference', model_dir='./', config=TestConfig())
# load coco model weights
rcnn.load_weights('mask_rcnn_void_cfg_00052.h5', by_name=True)
# load photograph

path=r'C:\Users\sai\Desktop\images\W1 (6).jpeg'
img = load_img(path)
img = img_to_array(img)
# make prediction
results = rcnn.detect([img], verbose=0)
# visualize the results
draw_image_with_boxes(path, results[0]['rois'])