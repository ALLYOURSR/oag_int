# Overview

A toy ML example designed to extract trends in CSV oil and gas data, built on tensorflow in python, with some help from numpy and scipy. Modular, reproducible ML, with parameterized, self-terminating training for semi-autonomous iteration.

# Recruiters

Hello! The primary purpose of this sample is to demonstrate familiarity with machine learning and coding in a dynamic language. I hope you'll find this code to be neat, organized, and ready to be collaboratively extended. Here is a sampling of specific examples that may interest you:

| File | Description |
| ---  | --- |
| NeuralNetFactory.py | Clean, idiomatic use of tensorflow |
| HeaderTypes.py | Abstractions to reduce dynamic language pitfalls (e.g. using an enum to prevent silent failure from mispelled strings) |
| objects module | OOP for improved readability and extensibility |
| requirements.txt | Pythonic practices for efficient code sharing |
| NeuralNetFactory.py | Understanding of neural network design (e.g. batch normalization, nonlinear activations, multiple layers) |
| All | Organization of code into submodules by function | 
| All | Readability and comments where necessary for optimized collaboration |


# Operation
The code is designed to parse column header CSV data files and train neural networks to predict values in a specified column. Comma delimited values belonging to columns specified by relevant headers (e.g. mcf_gas, bbls_oil_cum) are extracted and organized into a single 2D numpy array for optimized training in tensorflow. All execution is controlled by instances of the RunParams class in config/runparams.py partly shown below:

```python
class RunParams:
    def __init__(self, run_name, dict_params=None):
        default_vals = {
            'records_dir': "../run_records/",
            'data_dir': "../data/",
            'run_name': run_name,
            'num_neurons_per_layer': 8,
            'batch_size': 200,
            'num_training_steps': None,
            'neural_net_type': NeuralNetTypes.Basic,
            'train_rate': .1,
            'log_period': 100,

            'headers_to_evaluate': [
                                    HeaderTypes.lateral_length,
                                    HeaderTypes.stages,
									
									#Last header in this list is the value to predict
                                    HeaderTypes.cum_365_prod
									]
			
       ...
						}
```

Inspired by academic best practices, these parameters are written to disk prior to any training run, alongside training run results. In this manner, three goals are accomplished:

1. There is a central location in the code from which all execution can be conveniently controlled
2. Multiple training runs can be trivially scripted by initializing a list of RunParams instances with different parameters
3. Training runs are easily reproducable, and may be revisited weeks or months later by previously uninvolved developers


### Neural Network
For this exercise, I experimented with a number of modifications to a simple neural network<sup>1</sup> designed for general multivariable function approximation (neural_net/model_building.py). The unmodified net consists of a single, 8 neuron hidden layer with tan-sigmoid activations for nonlinearization. Training consisted of gradient descent minimization of root mean square error (RMSE). Trials were conducted with varied neuron counts, multiple hidden layers, and batch normalization, all recorded in the folder run_records. All networks performed with similar accuracy on the proprietary dataset (which I've since scrubbed from the commit history, NDA), although adding a batch normalization layer significantly reduced the number of training steps to convergence (I can't remember for sure, but something like 10x fewer steps).

### Automated Training Termination
I'm using a quick and dirty measure of long term slope of training error vs timestep. I've noticed in this example (and in other personal ML projects) that after ~10<sup>5</sup> training steps, a clear e<sup>-x</sup> shaped trend tends to emerge, and is consistent over ~10<sup>6</sup> training steps. After a couple hundred thousand or so steps the curve tends to flatten out as training converges. 

If one were monitoring MSE for convergence, a visually "flat enough" error curve would indicate that no further learning is likely to occur, and we've either found a global minimum, or a local minimum that we are unlikely to stumble out of with gradient descent. So to automate training termination, I simply calculate the slope of the error curve over some large number of training steps (10k in this case) and at some point because of the noisy nature of training error, the slope will be greater than 0, and I can terminate the training. Caveat: the width over which I calculate slope is going to vary with your particular dataset and neural net and will require some tweaking. This is a little handwavey, but in good hacker fashion, it works, so I use it!

# Random Notes
Training is probably either data transfer bound (RAM to GPU) or CPU bound, as GPU usage is rather low (~5%) compared with other ML that I've run on images in large convolutional neural nets (Inception V3 scale), which quickly become GPU bound and necessitate 100% fan speed.


# Key Themes


# TODO: Randomize numerical data and change column headers.

The two entry scripts are oag_int/main_data_engineering.py and oag_int/main_ml.py

1. Code Structure: rather than a series of nebulous scripts, I've architected a modular, extensible, object-oriented solution. For example, the class NeuralNetFactory (oag_int/objects/NeuralNetFactory.py) instantiates various neural nets which can then be trained and reused throughout the code. The code takes as input parameters (see oag_int/config/runparams.py) which are written to files before training runs to record conditions of past trials (neural net parameters, which columns to train in, etc) for iterative experimentation. Hopefully the structure is simple and intuitive enough to parse, but I can of course offer clarification as needed.
   


Bonus (oag_int/main_test.py): To loosely verify the function of the neural nets that I built for this exercise, I train and evaluate them on a noisy, multivariable polynomial with adjustable gaussian noise. The RMSE residuals reliably converge to the width of the noise function, suggesting some degree of validity.

Key points:
Modularity
Experimental Reproducability

# Data Parsing

# ML
### Neural Network Architectures and Factory

# Sources
1. Hagan, M. T.; Demuth, H. B.; Beale, M. H.; De Jesus, O. "Neural Network Design", 2nd ed, 2017. ISBN: 9780971732117

