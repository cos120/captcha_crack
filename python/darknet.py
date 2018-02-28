# -*- coding: UTF-8 -*-
from ctypes import *
import math
import random
import sys
import numpy as np
import itertools
import glob
import collections
import cPickle

import operator

sys.path.append('./')


def sample(probs):
    s = sum(probs)
    probs = [a / s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs) - 1


def c_array(ctype, values):
    return (ctype * len(values))(*values)


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib = CDLL("darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

make_boxes = lib.make_boxes
make_boxes.argtypes = [c_void_p]
make_boxes.restype = POINTER(BOX)

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

num_boxes = lib.num_boxes
num_boxes.argtypes = [c_void_p]
num_boxes.restype = c_int

make_probs = lib.make_probs
make_probs.argtypes = [c_void_p]
make_probs.restype = POINTER(POINTER(c_float))

detect = lib.network_predict
detect.argtypes = [
    c_void_p,
    IMAGE,
    c_float,
    c_float,
    c_float,
    POINTER(BOX),
    POINTER(
        POINTER(c_float))]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

network_detect = lib.network_detect
network_detect.argtypes = [
    c_void_p,
    IMAGE,
    c_float,
    c_float,
    c_float,
    POINTER(BOX),
    POINTER(
        POINTER(c_float))]


def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


def yolo_detect(net, meta, image, thresh=.4, hier_thresh=.4, nms=.45):
    boxes = make_boxes(net)
    probs = make_probs(net)
    num = num_boxes(net)
    network_detect(net, image, thresh, hier_thresh, nms, boxes, probs)
    res = []
    for j in range(num):
        for i in range(meta.classes):
            if probs[j][i] > 0:
                # print probs[j][i]
                res.append(
                    (meta.names[i],
                     probs[j][i],
                        (boxes[j].x,
                         boxes[j].y,
                         boxes[j].w,
                         boxes[j].h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(image)

    free_ptrs(cast(probs, POINTER(c_void_p)), num)
    return res


class Yolo():
    '''
    wrap darknet into python
    '''

    def __init__(self,net_cfg_path,net_weight,net_meta):
        self.net = load_net(net_cfg_path,net_weight,0)
        self.meta = load_meta(net_meta)

        #self.net = load_net("darknet/cfg/yolo-valid.cfg",
        #                    "darknet/model/yolo-weight.weights", 0)
        #self.meta = load_meta("darknet/cfg/yolo-origin.data")

    def detect(self, file):
        '''
        :param file:
        :type  Structure IMAGE
        :return:
        '''
        return yolo_detect(self.net, self.meta, file)


class Classifier():
    '''
    wrap darknet into python
    '''

    def __init__(self,net_cfg_path,net_weight,net_meta,chinese_dict):

        self.character_code = cPickle.load(
            open(chinese_dict, 'r'))
        self.code_character = {v: k for (k, v) in self.character_code.iteritems()}

        self.net = load_net( net_cfg_path, net_weight, 0)
        self.meta = load_meta(net_meta)

        #self.character_code = cPickle.load(
        #    open('darknet/model/character.dict', 'r'))
        #self.code_character = {v: k for (k, v) in self.character_code.iteritems()}

        #self.net = load_net(
        #    "darknet/cfg/chinese_character.cfg",
        #    "darknet/model/chinese_character.weights",
        #    0)
        #self.meta = load_meta("darknet/cfg/chinese.data")

    def code_to_character(self, code):
        return self.code_character[code]

    def character_to_code(self, character):
        return self.character_code[unicode(character, 'utf-8')]

    def classify(self, file, top=1000):
        '''
        classify image in top 100
        :param file: darknet input
        :type Structure IMAGE
        :param top: top 100 probability
        :return {character -> topk probability}
        :type dict
        '''
        # im = load_image(file, 0, 0)
        t = classify(self.net, self.meta, file)
        # result = collections.OrderedDict()
        result = {}

        for i in t[:top]:
            result[self.code_to_character(i[0])] = i[1]
        # result = sorted( result.items(), key=operator.itemgetter( 1 ) ,reverse=True)
        return result


def image_acc(path, top):
    classifier = Classifier()
    acc = 0
    total = len(path)
    top_1 = set()
    top_5 = set()
    for img in path:
        im = load_image(img, 0, 0)
        label = img[-11:-4]
        t = classifier.classify(im, top)
        # print label , [classifier.decode(x[0]) for x in t]
        top5 = [classifier.code_to_character(x[0]) for x in t]
        top_1.add(top5[0])
        top_5 = top_5.union(top5)
        if label in [classifier.code_to_character(x[0]) for x in t]:
            acc += 1
            continue
    return acc / total, top_1, top_5


if __name__ == "__main__":
    net = load_net("darknet/cfg/yolo-valid.cfg",
                   "darknet/yolo-weight.weights", 0)
    meta = load_meta("darknet/cfg/yolo-origin.data")
    im = load_image('test.jpeg', 0, 0)
    r = yolo_detect(net, meta, im)
