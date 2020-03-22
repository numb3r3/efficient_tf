"""Tensorflow graph related utilities."""

import logging
import tensorflow as tf
from tensorflow.python.framework import graph_util


def load_frozen_graph(frozen_graph_filename):
    """load a graph from protocol buffer file"""
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as in_f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(in_f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:  # pylint: disable=not-context-manager
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None)

    return graph


def load_graph_session_from_ckpt(ckpt_path, sess_config):
    """load graph and session from checkpoint file"""
    graph = tf.Graph()
    with graph.as_default():  # pylint: disable=not-context-manager
        sess = tf.Session(config=sess_config)
        with sess.as_default():  # pylint: disable=not-context-manager
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(ckpt_path))
            saver.restore(sess, ckpt_path)

    return graph, sess


def load_graph_session_from_pb(pb_file, sess_config):
    """load graph and session from protocol buffer file"""
    graph = load_frozen_graph(pb_file)
    with graph.as_default():
        sess = tf.Session(config=sess_config)
    return graph, sess


def frozen_graph_to_pb(outputs, frozen_graph_pb_path, sess, graph=None):
    """Freeze graph to a pb file."""
    if not isinstance(outputs, (list)):
        raise ValueError(
            "Frozen graph: outputs must be list of output node name")

    if graph is None:
        graph = tf.get_default_graph()

    input_graph_def = graph.as_graph_def()
    logging.info("Frozen graph: len of input graph nodes: {}".format(
        len(input_graph_def.node)))

    # We use a built-in TF helper to export variables to constant
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        outputs,
    )

    logging.info("Frozen graph: len of output graph nodes: {}".format(
        len(output_graph_def.node)))  # pylint: disable=no-member

    with tf.gfile.GFile(frozen_graph_pb_path, "wb") as in_f:
        in_f.write(output_graph_def.SerializeToString())
