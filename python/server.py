# -*- coding: utf-8 -*-

"""
author: zj
file: server.py
time: 17-11-3
"""
from ctypes import *
import os
import sys
from flask import Flask, request
from darknet import Yolo, IMAGE, Classifier
import cv2

import numpy as np
import scipy.misc
import urllib
import time
import json

reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('..')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_folder="imgs")
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


''':param global'''
bezel = 5 # yolo_split character bezel, default is 5, make character has high area rate of total image
darknet_input_pointer = None
darknet_input = None
# NOTE you chould use your model 
yolo_params = []
classifier_params = []
yolo = Yolo(*yolo_params)
classifier = Classifier(*classifier_params)
html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>image</h1>
    <form method=post enctype=multipart/form-data>
         character order<br>
         <input type=text name=text>
         <br>
         url<br>
         <input type=text name=url>
    </form>
    '''


def process_stream_to_image(img_stream):
    '''
    encode image from internet, stream -> 3d numpy array
    :param img_stream: stream file
    :return: image in numpy array
    :type numpy 3d array
    '''
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imdecode(img_array, cv2.CV_LOAD_IMAGE_COLOR), cv2.COLOR_BGR2RGB)

    return np.asarray(img)


def process_img_to_darknet_input(img):
    '''
    process image to darknet input format, 3d numpy array -> ctypes.c_float* array
    :param img: image as numpy array
    :return: darknet input
    :type ctypes.Structure IMAGE
    '''
    # BUG ctype pointer value changed when this function exits
    shape = img.shape
    img_input = np.asarray(img / 255., dtype=c_float)
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input.flatten().astype(c_float, casting='safe')
    p = img_input.ctypes.data_as(POINTER(c_float))
    global darknet_input
    darknet_input = IMAGE(c_int(shape[1]), c_int(shape[0]), c_int(shape[2]), p)
    return darknet_input

def process_stream_to_darknet_input(img_stream):

    img = process_stream_to_image(img_stream)
    return process_img_to_darknet_input(img)

def draw_yolo(img, meta_data):
    '''
    draw rectangle in origin image
    :param img: image input in 3d numpy array
    :param meta_data: yolo detection data, format (x_centre, y_centre, x_length, y_length)
    :return: image with rectangle labeled
    '''
    for data in meta_data:
        if data[1] > 0.3:
            xmin = data[2][0] - data[2][2] / 2.
            ymin = data[2][1] - data[2][3] / 2.
            xmax = data[2][0] + data[2][2] / 2.
            ymax = data[2][1] + data[2][3] / 2.
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 3)
    return img

def split_yolo(img, meta_data, shape):
    '''
    grab character by meta_data, which meta_data is calculated by yolo net
    meta format (x_centre, y_centre, x_length, y_length)
    :param img: 3d numpy image array
    :param meta_data: yolo net output
    :return: dictionary like {character:image array}
    '''
    images = {}
    for data in meta_data:
        if data[1] > 0.3:
            xmin = data[2][0] - data[2][2] / 2.
            ymin = data[2][1] - data[2][3] / 2.
            xmax = data[2][0] + data[2][2] / 2.
            ymax = data[2][1] + data[2][3] / 2.
            r = scipy.misc.imresize( img[ max( 0, int( ymin - bezel ) ):min( int( ymax + bezel ), shape[0 ] ),
                                     max( 0, int( xmin - bezel ) ):min( int( xmax + bezel ), shape[1 ] ) ],
                                     (64, 64) )
            images[(data[2][0], data[2][1])] = r
    return images

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def crack(meta, img, sequence, shape):
    '''
    crack captcha, use yolo meta data to grab image which contains character,
    classifier character images and locate each image position.
    return position in order
    :param meta: yolo net output
    :param img: origin image
    :param sequence: character sequence present captcha click order
    :return: click position order
    :type list of tuple, each tuple contains x and y
    '''
    character_count = len(sequence)
    images = split_yolo(img, meta, shape) # classifier input
    character_probs = {}
    for key, character_image in images.iteritems():
        shape = character_image.shape
        img_input = np.asarray(character_image / 255., dtype=c_float)
        img_input = np.transpose(img_input, (2, 0, 1))
        img_input = img_input.flatten().astype(c_float, casting='safe')
        darknet_input_pointer = img_input.ctypes.data_as(POINTER(c_float))
        darknet_input = IMAGE(
            c_int(
                shape[1]), c_int(
                shape[0]), c_int(
                shape[2]), darknet_input_pointer)
        # generate classifier output, dict{position->character probability}
        character_probs[key] = classifier.classify(darknet_input)
    # select probs which character appear in sequence
    positions_character_probs = [[probs.get(s, 0) for s in sequence]
                                 for position, probs in character_probs.iteritems()]
    positions = character_probs.keys()
    probs_array = np.asarray(positions_character_probs, dtype=np.float32)
    result_dict = {}
    for i in range(character_count): #non maximum suppression
        t = int(np.argmax(probs_array))
        x = t // character_count
        y = t % character_count
        result_dict[sequence[y]] = map(int, positions[x])
        probs_array[x, :] = -1
        probs_array[:, y] = -1

    return [result_dict[s] for s in sequence]
@app.route('/', methods=['GET'])
def hello():
    return 'hello'

@app.route('/post', methods=['GET','POST'])
def crack_by_post():
    '''
    :param text: character sequence
    :param url: image address in internet
    :return: template
    '''
    t = ['1', '2', '3']
    font = cv2.FONT_HERSHEY_SIMPLEX
    start = time.time()
    try:
        if request.method == 'POST':
            text1 = unicode(str(request.form['text']))
            text = [text1[0], text1[1], text1[2]]
            url = str(request.form.get('url', ''))
            if url: # load image from address
                req = urllib.urlopen(url)
                ## process image into darknet input format
                # BUG call function process_img_to_darknet_input return error,ctypes array return wrong value
                img_array = np.asarray(bytearray(req.read()), dtype=np.uint8)
                img = cv2.cvtColor(
                    cv2.imdecode(
                        img_array,
                        cv2.CV_LOAD_IMAGE_COLOR),
                    cv2.COLOR_BGR2RGB)
                shape = img.shape
                img_input = np.asarray(img / 255., dtype=c_float)
                img_input = np.transpose(img_input, (2, 0, 1))
                img_input = img_input.flatten().astype(c_float, casting='safe')
                darknet_input_pointer = img_input.ctypes.data_as(POINTER(c_float))
                darknet_input = IMAGE(c_int(shape[1]), c_int(shape[0]), c_int(shape[2]),
                                      darknet_input_pointer)
                ## process image into darknet input format
                meta = yolo.detect(darknet_input)
                if len(meta) <= len(text):
                    return_result['error'] = 1
                positions = crack( meta, img, text, shape )
                s = []
                for i in positions:
                    s.append(str(i[0]) + ',' + str(i[1]))
                return_result['result'] = ';'.join(s)
                return json.dumps(return_result)
        return html
    except BaseException:
        return html


if __name__ == '__main__':
    # app.run('0.0.0.0') # debug
    app.run() # gunicorn
    # test()
