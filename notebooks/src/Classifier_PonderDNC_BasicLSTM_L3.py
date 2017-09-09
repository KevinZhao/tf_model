import os
import os.path

import tensorflow as tf
import numpy as np

from tensorflow.python.saved_model import builder as saved_model_builder

from FixPonderDNCore import DNCore_L3
from FixPonderDNCore import ResidualACTCore as ACTCore


tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

class Classifier_PonderDNC_BasicLSTM_L3(object):

  def __init__(self, 
              inputs, 
              targets,
              gather_list=None,
              mini_batch_size=1, 
              hidden_size=10, 
              memory_size=10, 
              threshold=0.99,
              pondering_coefficient = 1e-2,
              num_reads=3,
              num_writes=1,
              learning_rate = 1e-4,
              optimizer_epsilon = 1e-10,
              max_gard_norm = 50):
        
    self._tmp_inputs = inputs
    self._tmp_targets = targets
    self._in_length = None
    self._in_width = inputs.shape[2]
    self._out_length = None
    self._out_width = targets.shape[2]
    self._mini_batch_size = mini_batch_size
    self._batch_size = inputs.shape[1]
    self._writer = None
    self._merged_summary = None
    self._remaining_iter = None
        
    # 声明计算会话
    self._sess = tf.InteractiveSession()
    
    self._inputs = tf.placeholder(dtype=tf.float32, 
                                  shape=[self._in_length, self._batch_size, self._in_width], 
                                  name='inputs')

    self._targets = tf.placeholder(dtype=tf.float32, 
                                   shape=[self._out_length, self._batch_size, self._out_width],
                                   name='targets')
    
    act_core = DNCore_L3( hidden_size=hidden_size, 
                          memory_size=memory_size, 
                          word_size=self._in_width, 
                          num_read_heads=num_reads, 
                          num_write_heads=num_writes)       

    self._InferenceCell = ACTCore(core=act_core, 
                                  output_size=self._out_width, 
                                  threshold=threshold, 
                                  get_state_for_halting=self._get_hidden_state)
    
    self._initial_state = self._InferenceCell.initial_state(self._batch_size)
    
    tmp, act_final_cumul_state = \
    tf.nn.dynamic_rnn(cell=self._InferenceCell, 
                      inputs=self._inputs, 
                      initial_state=self._initial_state, 
                      time_major=True)

    act_output, (act_final_iteration, act_final_remainder) = tmp
    '''
    inputs:
    If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements.

    If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.
    outputs:
    If time_major == False (default), this will be a `Tensor` shaped:
      `[batch_size, max_time, cell.output_size]`.

    If time_major == True, this will be a `Tensor` shaped:
      `[max_time, batch_size, cell.output_size]`.
    '''
        
    # 测试
    self._final_iteration = tf.reduce_mean(act_final_iteration)
    
    self._act_output = act_output

    if gather_list is not None:
        out_sequences = tf.gather(act_output, gather_list)
    else:
        out_sequences = act_core
    
    # 设置损失函数
    pondering_cost = (act_final_iteration + act_final_remainder) * pondering_coefficient
    rnn_cost = tf.nn.softmax_cross_entropy_with_logits(
        labels=self._targets, logits=out_sequences)
    self._pondering_cost = tf.reduce_mean(pondering_cost)
    self._rnn_cost = tf.reduce_mean(rnn_cost)
    self._cost = self._pondering_cost + self._rnn_cost        
    self._pred = tf.nn.softmax(out_sequences, dim=2)
    correct_pred = tf.equal(tf.argmax(self._pred,2), tf.argmax(self._targets,2))
    self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    
    tf.summary.scalar("accuracy", self._accuracy)
    tf.summary.scalar("cost", self._cost)
    tf.summary.scalar("rnn_cost", self._rnn_cost)
    tf.summary.scalar("pondering_cost", self._pondering_cost)
    tf.summary.scalar("final_iteration", self._final_iteration)
    
    
    # 设置优化器
    # Set up optimizer with global norm clipping.
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(self._cost, trainable_variables), max_gard_norm)

    global_step = tf.get_variable(
          name="global_step",
          shape=[],
          dtype=tf.int64,
          initializer=tf.zeros_initializer(),
          trainable=False,
          collections = [tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate = learning_rate, epsilon = optimizer_epsilon)

    self._train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step = global_step)
  
    self._merged_summary = tf.summary.merge_all()
    self._writer = tf.summary.FileWriter('summary', self._sess.graph)
        
        
  # 待处理函数
  def _get_hidden_state(self, state):
      controller_state, access_state, read_vectors = state
      layer_1, layer_2, layer_3 = controller_state
      L1_next_state, L1_next_cell = layer_1
      L2_next_state, L2_next_cell = layer_2
      L3_next_state, L3_next_cell = layer_3
      return tf.concat([L1_next_state, L2_next_state, L3_next_state], axis=-1)

  # 存储模型
  def _export_model(self, save_path):

      export_path = save_path

      export_path_base = save_path
      export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))

      #Model save for tensorflow serving
      builder = saved_model_builder.SavedModelBuilder(export_path)

      '''
      classification_inputs = tf.saved_model.utils.build_tensor_info(self._inputs)
      #print(self._inputs.shape, self._targets.shape)
      classification_outputs_classes = tf.saved_model.utils.build_tensor_info(self._targets)
      classification_outputs_scores = tf.saved_model.utils.build_tensor_info(self._act_output)

      classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={
                  tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                      classification_inputs
              },
              outputs={
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                      classification_outputs_classes
                  #tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                      #classification_outputs_scores
              },
              method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
      '''

      tensor_info_inputs = tf.saved_model.utils.build_tensor_info(self._inputs)
      tensor_info_outputs = tf.saved_model.utils.build_tensor_info(self._targets)

      prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'stock': tensor_info_inputs},
              outputs={'scores': tensor_info_outputs},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

      legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
      builder.add_meta_graph_and_variables(
          self._sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={
              'predict_stock':
                  prediction_signature#,
              #tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              #    classification_signature,
          },
          legacy_init_op=legacy_init_op)

      builder.save()
  
  def fit(
    self, 
    training_iters =1e2,             
    display_step = 5, 
    save_path = None,
    export_path = None,
    restore_path = None):

    self._sess.run(tf.global_variables_initializer())

    # 保存和恢复
    self._variables_saver = tf.train.Saver()

    if restore_path is not None:
      self._variables_saver.restore(self._sess, restore_path)

    if self._batch_size == self._mini_batch_size:

      for scope in range(np.int(training_iters)):

        _, loss, acc, tp1, tp2, tp3, summary= \
            self._sess.run([self._train_step, 
                            self._cost, 
                            self._accuracy, 
                            self._pondering_cost, 
                            self._rnn_cost, 
                            self._final_iteration,
                            self._merged_summary
                            ],
                            feed_dict = {
                                  self._inputs:self._tmp_inputs, 
                                  self._targets:self._tmp_targets
                                  })

        self._writer.add_summary(summary, scope)
        # 显示优化进程
        
        if scope % display_step == 0:
          print (scope, 
                 '  loss--', loss, 
                 '  acc--', acc, 
                 '  pondering_cost--',tp1, 
                 '  rnn_cost--', tp2, 
                 '  final_iteration', tp3)

          # 保存模型可训练变量
          if save_path is not None:
            self._variables_saver.save(self._sess, save_path)

        self._remaining_iter = training_iters -1 - scope
        pass

      print ("Optimization Finished!")
      self._export_model(export_path)

    else:
        print ('未完待续')

  def remaining_iter(self):
      return self._remaining_iter

  def close(self):
      self._sess.close()
      print ('结束进程，清理tensorflow内存/显存占用')

  def load_model(self, restore_path):
      if restore_path is not None:
          self._sess.run(tf.global_variables_initializer())        
          self._variables_saver = tf.train.Saver()
          self._variables_saver.restore(self._sess, restore_path)
          print('model get restored at', restore_path)

      
  def pred(self, inputs, gather_list=None, restore_path=None):
      if restore_path is not None:
          self._sess.run(tf.global_variables_initializer())        
          self._variables_saver = tf.train.Saver()
          self._variables_saver.restore(self._sess, restore_path)
          
      output_pred = self._act_output

      if gather_list is not None:
          output_pred = tf.gather(output_pred, gather_list)

      probability = tf.nn.softmax(output_pred)
      classification = tf.argmax(probability, axis=-1)
      
      return self._sess.run([probability, classification], feed_dict = {self._inputs:inputs})