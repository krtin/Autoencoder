"""
    This is the main model file for autoencoder
    it contains graphs, train, val and decode operations, placeholders
    Currently the model is a simple autoencoder which learns a 256 dimensional hidden representation

"""

import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS


class autoencoder(object):

    def __init__(self, mode, batch_size, nfeatures):
        self._mode = mode
        self._batch_size = batch_size
        self._nfeatures = nfeatures

    def _add_placeholders(self):

        self._input_batch = tf.placeholder(tf.float32, [self._batch_size, self._nfeatures], name='input_batch')
        self._target_batch = tf.placeholder(tf.float32, [self._batch_size, self._nfeatures], name='target_batch')


    def _make_feed_dict(self, batch):

      feed_dict = {}
      #input is corrupted and target is original image
      target_batch, input_batch = zip(*batch)
      feed_dict[self._input_batch] = input_batch
      feed_dict[self._target_batch] = target_batch


      return feed_dict

    def build_graph(self):
      """Add the placeholders, model, global step, train_op and summaries to the graph"""
      tf.logging.info('Building graph...')
      t0 = time.time()
      self._add_placeholders()

      with tf.device("/gpu:%d"%(FLAGS.sel_gpu)):
        self._add_autoencoder()

      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      if self._mode == 'train':
        self._add_train_op()
      self._summaries = tf.summary.merge_all()
      t1 = time.time()
      tf.logging.info('Time to build graph: %i seconds', t1 - t0)


    def run_train_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)


    def _add_train_op(self):

      # Take gradients of the trainable variables w.r.t. the loss function to minimize
      loss_to_minimize = self._loss
      tvars = tf.trainable_variables()
      gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

      # Clip the gradients
      with tf.device("/gpu:%d"%(FLAGS.sel_gpu)):
        grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)

      # Add a summary
      tf.summary.scalar('global_norm', global_norm)

      # Apply adagrad optimizer
      optimizer = tf.train.AdagradOptimizer(FLAGS.lr, initial_accumulator_value=FLAGS.adagrad_init_acc)
      with tf.device("/gpu:%d"%(FLAGS.sel_gpu)):
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def run_eval_step(self, sess, batch):

      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'summaries': self._summaries,
          'loss': self._loss,
          'global_step': self.global_step,
      }

      return sess.run(to_return, feed_dict)

    def run_decode_step(self, sess, batch):

      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'loss': self._loss
      }

      return sess.run(to_return, feed_dict)

    def _add_autoencoder(self):

        self.xavier_init = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

        with tf.variable_scope('autoencoder'):

            #encoder part
            if(FLAGS.freeze):
                self.bias_h = tf.get_variable('bias_hidden', [FLAGS.hidden_dim], dtype=tf.float32, trainable=False)
                self.enc_W = tf.get_variable('enc_W', [self._nfeatures, FLAGS.hidden_dim], dtype=tf.float32, initializer=self.xavier_init, trainable=False)
            else:
                self.bias_h = tf.get_variable('bias_hidden', [FLAGS.hidden_dim], dtype=tf.float32)
                self.enc_W = tf.get_variable('enc_W', [self._nfeatures, FLAGS.hidden_dim], dtype=tf.float32, initializer=self.xavier_init)
            self.encoded = tf.nn.sigmoid(tf.matmul(self._input_batch, self.enc_W) + self.bias_h)

            #decoder part
            self.bias_v = tf.get_variable('bias_visible', [self._nfeatures], dtype=tf.float32)
            self.dec_W = tf.get_variable('dec_W', [FLAGS.hidden_dim, self._nfeatures], dtype=tf.float32, initializer=self.xavier_init)
            self.decoded = tf.nn.sigmoid(tf.matmul(self.encoded, self.dec_W) + self.bias_v)


        with tf.name_scope("loss"):
            #cross entropy loss
            self._loss = - tf.reduce_sum(self._target_batch * tf.log(self.decoded))
            #mean square loss
            #self._loss = tf.sqrt(tf.reduce_mean(tf.square(self.input_data - self.decode)))
            tf.summary.scalar('loss', self._loss)


class rnnautoencoder(object):

    def __init__(self, mode, batch_size, nfeatures):
        self._mode = mode
        self._batch_size = batch_size
        self._nfeatures = nfeatures

    def _add_placeholders(self):

        self._input_batch = tf.placeholder(tf.int32, [self._batch_size, self._nfeatures], name='input_batch')
        self._target_batch = tf.placeholder(tf.int32, [self._batch_size, self._nfeatures+1], name='target_batch')
        self._dec_input_batch = tf.placeholder(tf.int32, [self._batch_size, self._nfeatures+1], name='dec_input_batch')
        self._padding_mask = tf.placeholder(tf.float32, [self._batch_size, self._nfeatures+1], name='dec_padding_mask')


    def _make_feed_dict(self, batch):

      feed_dict = {}
      #input is corrupted and target is original image
      dec_input_batch, target_batch, input_batch, mask_pad = zip(*batch)
      feed_dict[self._input_batch] = np.array(input_batch)
      feed_dict[self._dec_input_batch] = np.array(dec_input_batch)
      feed_dict[self._target_batch] = np.array(target_batch)
      feed_dict[self._padding_mask] = np.array(mask_pad)


      return feed_dict

    def build_graph(self):
      """Add the placeholders, model, global step, train_op and summaries to the graph"""
      tf.logging.info('Building graph...')
      t0 = time.time()
      self._add_placeholders()

      with tf.device("/gpu:%d"%(FLAGS.sel_gpu)):
        self._add_autoencoder()

      self.global_step = tf.Variable(0, name='global_step', trainable=False)
      if self._mode == 'train':
        self._add_train_op()
      self._summaries = tf.summary.merge_all()
      t1 = time.time()
      tf.logging.info('Time to build graph: %i seconds', t1 - t0)


    def run_train_step(self, sess, batch):
        feed_dict = self._make_feed_dict(batch)
        to_return = {
            'train_op': self._train_op,
            'summaries': self._summaries,
            'loss': self._loss,
            'global_step': self.global_step,
        }

        return sess.run(to_return, feed_dict)


    def _add_train_op(self):

      # Take gradients of the trainable variables w.r.t. the loss function to minimize
      loss_to_minimize = self._loss
      tvars = tf.trainable_variables()

      if(FLAGS.freeze):
          tvars = [var for var in tvars if "autoencoder/encoder" not in var.name]
          


      gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

      # Clip the gradients
      with tf.device("/gpu:%d"%(FLAGS.sel_gpu)):
        grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)

      # Add a summary
      tf.summary.scalar('global_norm', global_norm)

      # Apply adagrad optimizer
      optimizer = tf.train.AdagradOptimizer(FLAGS.lr, initial_accumulator_value=FLAGS.adagrad_init_acc)
      with tf.device("/gpu:%d"%(FLAGS.sel_gpu)):
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')

    def run_eval_step(self, sess, batch):

      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'summaries': self._summaries,
          'loss': self._loss,
          'global_step': self.global_step,
      }

      return sess.run(to_return, feed_dict)

    def run_decode_step(self, sess, batch):

      feed_dict = self._make_feed_dict(batch)
      to_return = {
          'loss': self._loss
      }

      return sess.run(to_return, feed_dict)

    def _add_decoder(self, inputs, cell):
        state = self.encoded_state
        outputs = []
        for i, inp in enumerate(inputs):
          tf.logging.info("Adding attention_decoder timestep %i of %i", i, len(inputs))
          if i > 0:
            variable_scope.get_variable_scope().reuse_variables()
          cell_output, state = cell(inp, state)

          outputs.append(cell_output)

        return outputs, state

    def _add_autoencoder(self):



        with tf.variable_scope('autoencoder'):

            self.rand_unif_init = tf.random_uniform_initializer(-0.02, 0.02, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=1e-4)

            with tf.variable_scope('encoder'):
                enc_embedding = tf.get_variable('enc_embedding', [16, FLAGS.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                emb_enc_inputs = [tf.nn.embedding_lookup(enc_embedding, x) for x in tf.unstack(self._input_batch, axis=1)]
                enc_cell = tf.contrib.rnn.LSTMCell(FLAGS.rnn_hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
                print('input', self._input_batch.shape)
                print('emb_enc_inputs', len(emb_enc_inputs), emb_enc_inputs[0].shape)

                (outputs, self.encoded_state) = tf.nn.static_rnn(enc_cell, emb_enc_inputs, dtype=tf.float32)

            with tf.variable_scope('decoder'):
              dec_embedding = tf.get_variable('dec_embedding', [17, FLAGS.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
              emb_dec_inputs = [tf.nn.embedding_lookup(dec_embedding, x) for x in tf.unstack(self._dec_input_batch, axis=1)]

              print('emb_dec_inputs', len(emb_dec_inputs), emb_dec_inputs[0].shape)

              dec_cell = tf.contrib.rnn.LSTMCell(FLAGS.rnn_hidden_dim, state_is_tuple=True, initializer=self.rand_unif_init)

              #create a custom RNN to add beam search later on
              decoder_outputs, self._dec_out_state = self._add_decoder(emb_dec_inputs, dec_cell)

              print("decoder output len", len(decoder_outputs))
              print("decoder output", decoder_outputs[0].shape)
              print("decoder state", self._dec_out_state.c.shape)


            with tf.variable_scope('output_projection'):
              vsize = 17
              w = tf.get_variable('w', [FLAGS.rnn_hidden_dim, vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
              w_t = tf.transpose(w)
              v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
              vocab_scores = []

              for i, output in enumerate(decoder_outputs):
                if i > 0:
                  tf.get_variable_scope().reuse_variables()
                vocab_scores.append(tf.nn.xw_plus_b(output, w, v))


              vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]


            with tf.variable_scope('loss'):
             loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
             batch_nums = tf.range(0, limit=FLAGS.batch_size) # shape (batch_size)
             for dec_step, dist in enumerate(vocab_dists):
               targets = self._target_batch[:,dec_step] # The indices of the target words. shape (batch_size)
               indices = tf.stack( (batch_nums, targets), axis=1) # shape (batch_size, 2)
               gold_probs = tf.gather_nd(dist, indices) # shape (batch_size). prob of correct words on this step
               losses = -tf.log(tf.clip_by_value(gold_probs,1e-10,1.0))
               loss_per_step.append(losses)

             self._loss = tf.reduce_mean(loss_per_step)


            tf.summary.scalar('loss', self._loss)
            #raise Exception('Test')
