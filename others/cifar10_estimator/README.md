## Status of this directory

The best working option is

- TF 1.5.0 (requires libcuda precisely 9.0 not 9.1)
- 'Experiment' API. New API with EvalSpec etc gets stuck indefinitely
- So here's what's working: 
    - for Training
        - start_single_node.sh, calling
        - ...cifar10_with_resnet_main.py
    - for Inference
        - cifar_resnet_inference.sh with
        - ...cifar_resnet_inference.py

So, the approach outlined in cifar10_new_resnet_main.py is not operational.
We keep the source files here for reference, though, maybe the problem is solved in a later version
