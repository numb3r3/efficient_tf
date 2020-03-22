"""The compatible tensorflow library."""

from tensorflow.compat.v1 import *

# Import absl.flags and absl.logging to overwrite the Tensorflow ones.
# This is the intended behavior in TF 2.0.
from absl import flags
from absl import logging
from tensorflow.python.compat import v2_compat

v2_compat.disable_v2_behavior()
# tf.disable_eager_execution()