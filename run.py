"""
    This is the main file which will be used to run the autoencoder model
    This file handles training, validation and testing/decoding
    Currently this only includes mnist dataset

    Requirements
    - Python3
    - Tensorflow >=1.6

    Basic Usage
      for training/validation
        - python run.py --exp_name=mymodel --mode=train
        - python run.py --exp_name=mymodel --mode=val (run this along with train concurrently)

      for decoding (Currently just calculates loss)
        - python run.py --exp_name=mymodel --mode=decode

    Additional Options
    - use --freeze=1 for freezing the decoder weights (after running freezing mode you won't be able to run non-freezing mode and as of now the saved model will have to be restored properly)
    - use --antimode=1 for using the anti-decoder model, which uses the opposite of mnist images
    - use --dataset=timeseries for using timeseries data default is mnist data


"""
import os
import time
import numpy as np
import tensorflow as tf
from model import autoencoder
from model import rnnautoencoder
from batcher import Batcher
import util


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'mnist', 'options ["mnist", "timeseries"]')
tf.app.flags.DEFINE_string('mode', '', 'train, val or decode')
tf.app.flags.DEFINE_string('exp_name', '', 'name of experiment')
tf.app.flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
tf.app.flags.DEFINE_string('mnist_dir', "MNIST_data/", 'Path to the cifar 10 dataset directory.')
tf.app.flags.DEFINE_string('err_type', 'masking', 'Type of input corruption "masking" or "salt_and_pepper"')
tf.app.flags.DEFINE_float('err_frac', 0.2, 'Fraction of the input to corrupt.')

tf.app.flags.DEFINE_boolean('train_encoder', True, 'Whether to learn encoder weights')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'restore best model')
#for freezing weights
tf.app.flags.DEFINE_boolean('freeze', False, 'Freeze training encoder weights')
#anti encoder
tf.app.flags.DEFINE_boolean('antimode', False, 'turn on/off anti-autoencoder mode')
tf.app.flags.DEFINE_integer('batch_size', 10, 'batch size')
tf.app.flags.DEFINE_integer('sel_gpu', 0, 'which gpu to use')
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of hidden units')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'embedding dimension for rnn auto encoder')
tf.app.flags.DEFINE_integer('rnn_hidden_dim', 128, 'rnn hidden dimension for rnn auto encoder')
tf.app.flags.DEFINE_float('lr', 0.01, 'learning rate')
tf.app.flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
tf.app.flags.DEFINE_string('optimizer', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

def restore_best_model():

  print("Restoring bestmodel for training...")

  # Initialize all vars in the model
  sess = tf.Session(config=util.get_config())
  print("Initializing all variables...")
  sess.run(tf.initialize_all_variables())

  # Restore the best model from eval dir
  saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
  print("Restoring all non-adagrad variables from best model in eval dir...")
  curr_ckpt = util.load_ckpt(saver, sess, "val")
  print("Restored %s." % curr_ckpt)


  new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
  new_fname = os.path.join(FLAGS.exp_name, "train", new_model_name)
  print("Saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver()
  new_saver.save(sess, new_fname)
  print("Saved.")
  exit()

def setup_training(model, batcher):
    train_dir = os.path.join(FLAGS.exp_name, "train")
    if not os.path.exists(train_dir): os.makedirs(train_dir)

    model.build_graph() # build the graph

    if FLAGS.restore_best_model:
      restore_best_model()

    saver = tf.train.Saver(max_to_keep=3)

    sv = tf.train.Supervisor(logdir=train_dir,
                       is_chief=True,
                       saver=saver,
                       summary_op=None,
                       save_summaries_secs=10, # save summaries for tensorboard every 60 secs
                       save_model_secs=10, # checkpoint every 60 secs
                       global_step=model.global_step)

    summary_writer = sv.summary_writer
    tf.logging.info("Preparing or waiting for session...")
    sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
    tf.logging.info("Created session.")


    try:
      run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
    except KeyboardInterrupt:
      tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
      sv.stop()

def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    tf.logging.info("starting run_training")
    with sess_context_manager as sess:

        batches = batcher.getBatches()
        noof_epochs = 0

        while(True):
            output_interval=100
            epoch_avg_loss = 0
            epoch_train_steps = 0
            sum=0.
            for batch in batches:

                #tf.logging.info('running training step...')
                t0=time.time()
                results = model.run_train_step(sess, batch)
                t1=time.time()
                #tf.logging.info('seconds for training step: %.3f', t1-t0)

                loss = results['loss']

                epoch_train_steps += 1
                #tf.logging.info('loss: %f', loss) # print the loss to screen
                epoch_avg_loss = (epoch_avg_loss*(epoch_train_steps-1.) + loss)/epoch_train_steps
                #if((epoch_train_steps-1)%output_interval==0):
                #    print('Avg loss for Epoch %d: %f, interval loss: %f'%(noof_epochs+1, epoch_avg_loss, float(sum/output_interval)))
                #    tf.logging.info('loss: %f', loss)
                #    sum=0.
                sum += loss
                if not np.isfinite(loss):
                  raise Exception("Loss is not finite. Stopping.")

                summaries = results['summaries']
                train_step = results['global_step']

                summary_writer.add_summary(summaries, train_step)
                if train_step % 100 == 0:
                  summary_writer.flush()

            noof_epochs += 1
            print("Final Average loss for Epoch %d: %f"%(noof_epochs, epoch_avg_loss))
            tf.logging.info('Epoch %d finished'%noof_epochs)

            if(FLAGS.dataset=='timeseries'):
                batcher.getData(updateData=True)
                batches = batcher.getBatches()


def run_eval(model, batcher):

  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.exp_name, "val") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far
  batches = batcher.getBatches()

  while True:

    _ = util.load_ckpt(saver, sess) # load a new checkpoint

    epoch_avg_loss = 0.
    epoch_train_steps = 0

    for batch in batches:

        # run eval on the batch
        t0=time.time()

        results = model.run_eval_step(sess, batch)

        t1=time.time()
        #tf.logging.info('seconds for batch: %.2f', t1-t0)

        # print the loss and coverage loss to screen
        loss = results['loss']
        epoch_train_steps += 1
        #tf.logging.info('loss: %f', loss)

        # add summodemaries
        summaries = results['summaries']
        train_step = results['global_step']
        summary_writer.add_summary(summaries, train_step)

        epoch_avg_loss = (epoch_avg_loss*(epoch_train_steps-1.) + loss)/epoch_train_steps


    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    print("Average loss for Epoch %f"%(epoch_avg_loss))
    if best_loss is None or epoch_avg_loss < best_loss:
      #tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      print('Found new best model with %f epoch_avg_loss. Saving to %s'% (epoch_avg_loss, bestmodel_save_path))
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = epoch_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()

    if(FLAGS.dataset=='timeseries'):
        batcher.getData(updateData=True)
        batches = batcher.getBatches()


def run_decoding(model, batcher):

  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  decode_dir = os.path.join(FLAGS.exp_name, "decode") # make a subdir of the root dir for eval data
  batches = batcher.getBatches()

  _ = util.load_ckpt(saver, sess) # load a new checkpoint

  epoch_avg_loss = 0.
  epoch_decode_steps = 0

  for batch in batches:
    results = model.run_decode_step(sess, batch)
    loss = results['loss']
    epoch_decode_steps += 1
    epoch_avg_loss = (epoch_avg_loss*(epoch_decode_steps-1.) + loss)/epoch_decode_steps

  print("Average loss %f"%(epoch_avg_loss))




def main(unused_argv):
    if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
      raise Exception("Problem with flags: %s" % unused_argv)
    if(FLAGS.mode!='train'):
        FLAGS.batch_size = 1

    batcher = Batcher(FLAGS.dataset, "MNIST_data/", FLAGS.mode, FLAGS.batch_size, FLAGS.err_type, FLAGS.err_frac, FLAGS.antimode)
    nfeatures = batcher.nfeatures

    if(FLAGS.dataset=='timeseries'):
        model = rnnautoencoder(FLAGS.mode, FLAGS.batch_size, nfeatures)
    else:
        model = autoencoder(FLAGS.mode, FLAGS.batch_size, nfeatures)

    if(FLAGS.mode=='train'):
        setup_training(model, batcher)
    elif(FLAGS.mode=='val'):
        run_eval(model, batcher)
    elif(FLAGS.mode=='decode'):
        run_decoding(model, batcher)



if __name__ == '__main__':
  tf.app.run()
