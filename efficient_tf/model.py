"""Abstract model class."""
import os

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework import graph_util

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils as saved_model_utils


class BaseModel(object):

    def __init__(self, args):
        self.args = args

        self.loss = None
        self.predictions = {}
        self.eval_metrics = {}    # other auxiliary eval metric

        self.saver = tf.train.Saver()

    def get_features(self, params):
        """Placeholder for models."""
        raise NotImplementedError

    def process_fn(self, mode, params, features, labels):
        raise NotImplementedError

    def model_fn(self, features, labels, mode, params):
        """Main model implementation"""
        raise NotImplementedError

    def restore_fn(self):
        pass

    def save(self, sess, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename + '.ckpt')
        self.saver.save(sess, filepath)
        return filepath

    def save_signature(self, sess, directory, params):

        signature = signature_def_utils.build_signature_def(
            # inputs={
            #     'input':
            #     saved_model_utils.build_tensor_info(self.input),
            #     'dropout_rate':
            #     saved_model_utils.build_tensor_info(self.dropout_rate)
            # },
            # outputs={
            #     'output': saved_model_utils.build_tensor_info(self.output)
            # },
            inputs={k: saved_model_utils.build_tensor_info(
                v) for k, v in self.get_features(params).items()},
            outputs={k: saved_model_utils.build_tensor_info(
                v)for k, v in self.predictions.items()},
            method_name=signature_constants.PREDICT_METHOD_NAME)
        signature_map = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
        }
        model_builder = saved_model_builder.SavedModelBuilder(directory)
        model_builder.add_meta_graph_and_variables(
            sess,
            tags=[tag_constants.SERVING],
            signature_def_map=signature_map,
            clear_devices=True)
        model_builder.save(as_text=False)

    def save_as_pb(self, sess, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save check point for graph frozen later
        ckpt_filepath = self.save(sess, directory=directory, filename=filename)
        pbtxt_filename = filename + '.pbtxt'
        pbtxt_filepath = os.path.join(directory, pbtxt_filename)
        pb_filepath = os.path.join(directory, filename + '.pb')
        # This will only save the graph but the variables will not be saved.
        # You have to freeze your model first.
        tf.train.write_graph(graph_or_graph_def=sess.graph_def,
                             logdir=directory,
                             name=pbtxt_filename,
                             as_text=True)

        # Freeze graph
        # Method 1
        freeze_graph.freeze_graph(input_graph=pbtxt_filepath,
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=ckpt_filepath,
                                  output_node_names='cnn/output',
                                  #   restore_op_name='save/restore_all',
                                  #   filename_tensor_name='save/Const:0',
                                  output_graph=pb_filepath,
                                  clear_devices=True,
                                  initializer_nodes='')

        # Method 2
        '''
        graph = tf.get_default_graph()
        input_graph_def = graph.as_graph_def()
        output_node_names = ['cnn/output']
        output_graph_def = graph_util.convert_variables_to_constants(self.sess, input_graph_def, output_node_names)
        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        '''

        return pb_filepath
