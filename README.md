# Neural-net-pruning
Semester project: Determine near-optimal pruning strategies for neural networks

Create a conda environment and activate it:

        conda env create -f environment.yml

Run the training script and launch the web-server:

        python train.py -cif 64 1000 10 Conv6 nn.CrossEntropyLoss optim.Adam lr=0.0003
        python visualize.py

`train.py` has the following arguments:
- `-cif`: if option is chosen, CIFAR-10 will be used instead of MNIST
- `train_batch_size`: an integer, e.g. 64
- `test_batch_size`: an integer, e.g. 1000
- `epochs`: number of training epochs, e.g. 10
- `model_class`: name of the class defined in `architecture.models`
- `criterion_class`: the training objective (has to start with _nn._)
- `optimizer_class`: the optimizer (has to start with _optim._)
- `optim_args`: list of float arguments given to the optimizer (e.g. _lr=0.001 momentum=0.9_)

The optimizer arguments will be stored in a list and must be key-word arguments.
