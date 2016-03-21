import model_input
import model_inference
import tensorflow as tf


def evaluate(test, batch_size):
    if (test):
        filenames = ['test.tfrecords']
    else:
        filenames = ['train.tfrecords']
    images, labels = model_input.regular_inputs(filenames, batch_size)

     logits = model_inference.inference(images, batch_size)

