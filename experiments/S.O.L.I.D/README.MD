## Principles and Patterns for ML geeks

##### By Wolfgang Giersche, Zühlke Engineering AG

#### S.O.L.I.D principles applied to Tensorflow programming

##### I am using the most current TF API 1.6.0 rc1 with the following building blocks:

- [Tensorflow Dataset API](https://www.tensorflow.org/programmers_guide/datasets)
    - Allows for pre-processing with a monadic API (map, flatmap, etc)
    - Preprocessing may even happen in parallel streaming fashion
    
- [Estimator API](https://www.tensorflow.org/programmers_guide/estimators)
    - very convenient highlevel API
    - Checkpointing and recovery 
    - Tensorboard summaries
    - much more...
- [Multi-GPU Training of contrib.estimator package](https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/)
    - convenient wrapper to distribute training on any number of GPUs on a single machine
    - works by means of synchonous gradient averaging over parallel mini-batches

This tutorial is based on  [Google's official TF example](https://github.com/tensorflow/models/tree/master/official/mnist)
using [Zalando Research's Fashion Dataset](https://github.com/zalandoresearch/fashion-mnist)
instead of the typical [Handwritten Digits](http://yann.lecun.com/exdb/mnist/).

The [S.O.L.I.D. Principles](http://www.cvc.uab.es/shared/teach/a21291/temes/object_oriented_design/materials_adicionals/principles_and_patterns.pdf) 
are commonly attributed to [Robert C. Martin (Uncle Bob)](https://de.wikipedia.org/wiki/Robert_Cecil_Martin).

![Anatomy of an ML epic](Anatomy-of-an-experiment.png)

Exactly because data analytics and machine learning have rather exploratory traits, 
practices better support changes without endangering the quality of the code.



Supplemental material:

- parallel training on two or more GPUs
- The concept of Estimator and EstimatorSpec
- Datasets and functional monadic interfaces
- The concept of a computational graph -> tf.InteractiveSession() to the rescue
- Neural networks: concepts

More non-academic resources:
- [Michael Nielson's blog](http://neuralnetworksanddeeplearning.com/)
- [Chris Olah's blog](http://colah.github.io/)
- [Olah on topology](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
- [A quest for true understanding: Distill.pub](https://distill.pub/)

More academic:
- [Goodfellow's Deep Learning Book](http://www.deeplearningbook.org/)


Later on request

- A (not quite) naive glimpse on the current understanding of neural networks
