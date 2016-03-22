import model_input
import model_inference
import model_train
import tensorflow as tf
import math
import numpy as np
from datetime import datetime

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', 'eval',
                           """Directory where to write event logs.""")

tf.app.flags.DEFINE_string('checkpoint_dir', 'train',
                           """Directory where to read model checkpoints.""")

tf.app.flags.DEFINE_integer('eval_batch_size', 100,
                            """Number of images to process in a batch.""")

tf.app.flags.DEFINE_boolean('test', False,
                            """ Run eval on test set or training set.""")


def eval_once(saver, summary_writer, top_k_op, summary_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    	if ckpt and ckpt.model_checkpoint_path:
			# Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            if(FLAGS.test):
                num_iter = int(math.ceil(model_input.TEST_SET_SIZE / FLAGS.eval_batch_size))
            else:
                num_iter = int(math.ceil(model_input.TRAINING_SET_SIZE / FLAGS.eval_batch_size))

            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.eval_batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(test, batch_size, eval_dir):
    if (test):
        filenames = ['test.tfrecords']
    else:
        filenames = ['train.tfrecords']

    images, labels = model_input.regular_inputs(filenames, batch_size, test)

    logits = model_inference.inference(images, batch_size)

    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model_train.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    graph_def = tf.get_default_graph().as_graph_def()
    summary_writer = tf.train.SummaryWriter(eval_dir,
                                            graph_def=graph_def)

    eval_once(saver, summary_writer, top_k_op, summary_op)

def main(argv=None):
    if tf.gfile.Exists(FLAGS.eval_dir):
   	    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate(FLAGS.test, FLAGS.eval_batch_size,FLAGS.eval_dir)


if __name__ == '__main__':
    tf.app.run()
