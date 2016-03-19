import tensorflow as tf

IMAGE_SIZE = 128
TRAINING_SET_SIZE = 8000


def readRecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def generate_batch(image, label, min_queue_examples, batch_size):
    num_preprocess_threads = 16
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    tf.image_summary('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(filenames, batch_size):
    filename_queue = tf.train.string_input_producer(filenames)

    image, label = readRecords(filename_queue)

    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = tf.image.per_image_whitening(distorted_image)
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=10)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.7, upper=1.3)

    min_fraction_examples_in_queue = 0.4
    num_examples_per_epoch = TRAINING_SET_SIZE
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_examples_in_queue)

    return generate_batch(distorted_image, label,
                          min_queue_examples, batch_size)
