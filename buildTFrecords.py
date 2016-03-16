import os
from scipy import misc
import json
import tensorflow as tf
import numpy as np

IMGDIR = "/media/lucas/Data/Dev/pools/images-split/splits/"
DICTFILE = "dict.json"
TRAINFILE = "train.tfrecords"
TESTFILE = "test.tfrecords"

labeldict = {}

if(os.path.isfile(DICTFILE)):
    with open(DICTFILE, 'r') as inputfile:
        labeldict = json.load(inputfile)
imglist = os.listdir(IMGDIR)
np.random.shuffle(imglist)

traininglist = imglist[:8000]
testlist = imglist[8000:]


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


trainwriter = tf.python_io.TFRecordWriter(TRAINFILE)
print('writing training set, size ' + str(len(traininglist)))
for imgpath in traininglist:
    img = misc.imread(IMGDIR + imgpath)
    if(imgpath in labeldict):
        label = labeldict[imgpath]
        image_raw = img.tostring()
        rows = img.shape[0]
        cols = img.shape[1]
        depth = img.shape[2]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)
            }))
        trainwriter.write(example.SerializeToString())


testwriter = tf.python_io.TFRecordWriter(TESTFILE)
print('writing test set, size ' + str(len(testlist)))
for imgpath in testlist:
    img = misc.imread(IMGDIR + imgpath)
    if(imgpath in labeldict):
        label = labeldict[imgpath]
        image_raw = img.tostring()
        rows = img.shape[0]
        cols = img.shape[1]
        depth = img.shape[2]
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)
            }))
        testwriter.write(example.SerializeToString())
