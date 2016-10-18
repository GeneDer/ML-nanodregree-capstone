import tensorflow as tf
import numpy as np
import json
import time
import sys
import os
from scipy.misc import imread, imresize
from scipy.ndimage import interpolation
from PIL import Image, ImageDraw, ImageFont
import PIL.ExifTags

from train import build_forward
from utils import googlenet_load, train_utils
from utils.train_utils import add_rectangles

    
image_size = 28
num_labels = 10
def number_classification(region_of_interest):
    h1 = 1200
    h2 = 300
    h3 = 75

    graph = tf.Graph()
    with graph.as_default():
        # digit image data
        tf_test_data = tf.constant(region_of_interest)
    
        # 3 fully connected layers
        weights_layer1 = tf.Variable(tf.truncated_normal([image_size * image_size, h1], stddev=0.01))
        biases_layer1 = tf.Variable(tf.zeros([h1]))
        weights_layer2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.01))
        biases_layer2 = tf.Variable(tf.zeros([h2]))
        weights_layer3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.01))
        biases_layer3 = tf.Variable(tf.zeros([h3]))
        weights = tf.Variable(tf.truncated_normal([h3, num_labels], stddev=0.01))
        biases = tf.Variable(tf.zeros([num_labels]))

        # load variables
        saver = tf.train.Saver()

        # evaluation
        hidden = tf.nn.relu(tf.matmul(tf_test_data, weights_layer1) + biases_layer1)
        hidden = tf.nn.relu(tf.matmul(hidden, weights_layer2) + biases_layer2)
        hidden = tf.nn.relu(tf.matmul(hidden, weights_layer3) + biases_layer3)
        test_prediction = tf.nn.softmax(tf.matmul(hidden, weights) + biases)

    with tf.Session(graph=graph) as session:
        # load variables
        saver.restore(session, "data/classification_model.ckpt")
        
        # classify the digit
        pred = test_prediction.eval()
        return (1+np.argmax(pred))%10


# load tensorbox settings
hypes_file = 'overfeat_rezoom.json'
with open(hypes_file, 'r') as f:
    H = json.load(f)
def overFeat(image_data, image_data_gray):
    # placeholder for digit outputs
    digit_list = []

    # keep track of time
    init_time = time.time()
    
    # load tensorbox
    tf.reset_default_graph()
    googlenet = googlenet_load.init(H)
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), googlenet, 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, "data/overfeat_checkpint.ckpt")

        # run tensorbox
        feed = {x_in: image_data}
        (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
        new_img, rects = add_rectangles(H, [image_data], np_pred_confidences, np_pred_boxes,
                                        use_stitching=True, rnn_len=H['rnn_len'], min_conf=0.3)
        
        for rec in rects:
            # threshold boxes
            if rec.score > 0.1:
                # sometimes the box will be outside of images,
                # set them to 0 if they are outside
                if rec.x1 < 0:
                    rec.x1 = 0
                if rec.y1 < 0:
                    rec.y1 = 0

                # crop digit and resize to the input size of classification
                number = imresize(image_data_gray[int(rec.y1):int(rec.y2),
                                                  int(rec.x1):int(rec.x2)],
                                  (image_size, image_size))

                # normalize the cropped region
                region_of_interest = (number.astype(float) - 255 / 2) / 255

                # classify the digit
                num = number_classification(region_of_interest.reshape((-1, image_size* image_size)).astype(np.float32))

                # keep results
                digit_list.append((num, rec.x1, rec.y1, rec.x2, rec.y2))
            
    # return the time used and the result
    return time.time() - init_time, digit_list


def load_image(image_name):
    image_path = "images_input/" + image_name
    
    # load image
    im = Image.open(image_path)
    image_data = imread(image_path, mode = "RGB")
    image_height, image_width, image_depth = image_data.shape
    image_data_gray = imread(image_path, flatten = True)

    # try to rotate the image if needed
    try:
        exif_data = im._getexif()
        exif = {
            PIL.ExifTags.TAGS[k]: v
            for k, v in exif_data.items()
            if k in PIL.ExifTags.TAGS
        }
        ori = exif['Orientation']

        if ori == 3:
            image_data = interpolation.rotate(image_data, 180)
            image_data_gray = interpolation.rotate(image_data_gray, 180)
            im = im.rotate(180)
        elif ori == 6:
            image_data = interpolation.rotate(image_data, 270)
            image_data_gray = interpolation.rotate(image_data_gray, 270)
            im = im.rotate(270)
        elif ori == 8:
            image_data = interpolation.rotate(image_data, 90)
            image_data_gray = interpolation.rotate(image_data_gray, 90)
            im = im.rotate(90)
    except:
        print "No EXIF data"

    # resize image
    im = im.resize((640, 480))
    image_data_gray = imresize(image_data_gray, (480, 640))
    image_data = imresize(image_data, (480, 640))

    # get the detection and clasification results
    time_used, digit_list = overFeat(image_data, image_data_gray)

    # keep a counter for each number
    counter = [0]*10
    
    #draw boxes and output images
    for digit in digit_list:

        # increment counter
        counter[digit[0]] += 1

        # draw box and number
        draw = ImageDraw.Draw(im)
        draw.rectangle([digit[1], digit[2], digit[3], digit[4]])
        draw.text([digit[1] + 2, digit[2] + 2], str(digit[0]))            
        del draw

    # save image
    im.save("images_output/%s"%image_name)

    # return time used for the models and the counter
    return time_used, counter

def main(image_list):
    total_time_used = 0
    digits_result = {}
    
    # loop through each images in the list
    for image in image_list:
        time_used, counter = load_image(image)
        total_time_used += time_used
        digits_result[image] = counter
        print image+" complete"
        
    print "Average time used per image:", total_time_used/len(image_list)
    json.dump(digits_result, open('data/digits_result.json', 'w'))
        

if __name__ == "__main__":
    # if no argument provided, evaluate all images,
    # if a specific image provided, evaluate only on that image
    if len(sys.argv) == 1:
        image_list = []
        for image in os.listdir("images_input"):
            file_name_split = image.split('.')
            if len(file_name_split) == 2 and file_name_split[0] != '':
                image_list.append(image)
        main(image_list)
    elif len(sys.argv) != 2:
        print "Error: please supply correct argument"
    elif os.path.exists(os.getcwd()+'/images_input/'+sys.argv[1]):
        main([sys.argv[1]])
    else:
        print "Error: no such file"
        
