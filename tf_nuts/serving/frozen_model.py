"""Frozen Model for serving"""
from typing import List
import os
import sys
import logging
import tensorflow as tf
from ..graph import load_frozen_graph


class FrozenModel():
    '''FrozenModel'''

    def __init__(self, model_path: str, gpu_devices: str = None):
        '''
         model_path: ckpt dir or frozen_graph_pb path
         gpu_devices: list of gpu devices. e.g. '' for cpu, '0,1' for gpu 0,1
        '''
        self.init_session(model_path, gpu_devices)

    def init_session(self, model_path: str, gpu_devices: str):
        # The config for CPU usage
        config = tf.ConfigProto()
        if not gpu_devices:
            config.gpu_options.visible_device_list = ''
        else:
            config.gpu_options.visible_device_list = gpu_devices
            config.gpu_options.allow_growth = True

        # check model dir
        if os.path.isdir(model_path):
            self._graph = tf.Graph()

            # checkpoint
            self._sess = tf.Session(
                graph=self._graph,
                config=tf.ConfigProto(
                    allow_soft_placement=True, log_device_placement=True))
            ckpt_path = tf.train.latest_checkpoint(model_path)
            # self._graph, self._sess = utils.load_graph_session_from_ckpt(ckpt_path)
            model_path = ckpt_path + '.meta'
            logging.info("meta : {}".format(model_path))
            saver = tf.train.import_meta_graph(model_path)
            saver.restore(self._sess, ckpt_path)

        else:
            if not os.path.exists(model_path):
                logging.info('{}, is not exist'.format(model_path))
                logging.info("frozen_graph : {} not exist".format(model_path))
                sys.exit(0)

            # frozen graph pb
            frozen_graph = model_path
            logging.info('frozen graph pb : {}'.format(frozen_graph))
            self._graph = load_frozen_graph(frozen_graph)
            self._sess = tf.Session(graph=self._graph, config=config)

    @property
    def graph(self):
        return self._graph

    @property
    def sess(self):
        return self._sess
