import collections

import six
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter


# TODO(b/64848083) Remove once uid bug is fixed
# noinspection PyUnresolvedReferences
class RunConfig(tf.contrib.learn.RunConfig):
    def uid(self, whitelist=None):
        """Generates a 'Unique Identifier' based on all internal fields.
        Caller should use the uid string to check `RunConfig` instance integrity
        in one session use, but should not rely on the implementation details, which
        is subject to change.
        Args:
          whitelist: A list of the string names of the properties uid should not
            include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
            includes most properties user allowes to change.
        Returns:
          A uid string.
        """
        if whitelist is None:
            # noinspection PyProtectedMember
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        # Pop out the keys in whitelist.
        for k in whitelist:
            state.pop('_' + k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # For class instance without __repr__, some special cares are required.
        # Otherwise, the object address will be used.
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(),
                       key=lambda t: t[0])
            )
        return ', '.join(
            '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))


def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops is None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        # noinspection PyProtectedMember
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def device_setter_fn(variable_strategy, worker_device, num_gpus):
    my_device_setter = None
    if variable_strategy == 'CPU':
        my_device_setter = local_device_setter(
            worker_device=worker_device)
    elif variable_strategy == 'GPU':
        my_device_setter = local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
    return my_device_setter
