def add_default_arguments(parser):
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.')
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='CPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--num-layers',
        type=int,
        default=44,
        help='The number of layers of the model.')
    parser.add_argument(
        '--train-steps',
        type=int,
        default=2000,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=128,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=100,
        help='Batch size for validation.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help="""\
          This is the inital learning rate value. The learning rate will decrease
          during training. For more details check the model_fn implementation in
          this file.\
          """)
    parser.add_argument(
        '--use-distortion-for-training',
        type=bool,
        default=False,
        help='If doing image distortion for training.')
    parser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help="""\
          If present when running in a distributed environment will run on sync mode.\
          """)
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
          Number of threads to use for intra-op parallelism. When training on CPU
          set to 0 to have the system pick the appropriate number or alternatively
          set it to the number of physical CPU cores.\
          """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
          Number of threads to use for inter-op parallelism. If set to 0, the
          system will pick an appropriate number.\
          """)
    parser.add_argument(
        '--data-format',
        type=str,
        default=None,
        help="""\
          If not set, the data format best for the training device is used. 
          Allowed values: channels_first (NCHW) channels_last (NHWC).\
          """)
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.997,
        help='Decay for batch norm.')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-5,
        help='Epsilon for batch norm.')
