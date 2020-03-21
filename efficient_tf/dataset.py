"""Base dataset class"""

import os
import logging
import tensorflow as tf


class BaseDataset:

    def __init__(self, args):
        self.args = args

    def prepare(self, params):
        """This function will be called once to prepare the dataset."""
        pass

    def read(self, split, params):
        """Create an instance of the dataset object."""
        return {
            "train": tf.data.TFRecordDataset(params['train_files']),
            "eval": tf.data.TFRecordDataset(params['eval_files']),
        }[split]

    def parse(self, mode, params, tf_record):
        """Parse input record to features and labels."""
        pass

    def process(self, mode, params, features, labels):
        """Parse input record to features and labels."""
        return features, labels
