import tensorflow as tf
import platform

def os_info():
    return {
        'machine': platform.machine(),
        'node': platform.node(),
        'os': platform.platform(),
        'cuda': tf.test.is_built_with_cuda()
    }
