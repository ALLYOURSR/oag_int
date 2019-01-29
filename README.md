# Overview

A toy ML example designed to extract trends in CSV oil and gas data, built on tensorflow in python, with some help from numpy and scipy. Modular, reproducible ML experimentation, with self-terminating training scripts for semi-autonomous iteration.

# Recruiters

Hello! Here's what I primarily hope you gather from this page: I have experience in dynamic languages, I am able to use their advantages. I'm familiar with ML, use it reasonably, I write organized code.

# Operation
The code is designed to parse column header CSV data files and train neural networks to predict a specified column. Comma delimited values belonging to columns specified by relevant headers (e.g. mcf_gas, bbls_oil_cum) are extracted and organized into a single 2D numpy array for optimized training in tensorflow. All execution is controlled by instances of the RunParams class in config/runparams.py partly shown below:

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
        }
```

Inspired by academic research best practices, these parameters are written to disk prior to any training run, alongside training run results. In this manner, three goals are accomplished:

1. There is a central location in the code from which all execution can be conveniently controlled
2. Multiple training runs can be trivially scripted by initializing a list of RunParams instances with different parameters
3. Training runs are easily reproducable, and may be revisited weeks or months later


### Neural Net
For this exercise, I experimented with a number of modifications to a simple neural network designed for general multivariable function approximation (oag_int/neural_net/model_building.py)

### Automated Training Termination




# Key Themes


The two entry scripts are oag_int/main_data_engineering.py and oag_int/main_ml.py

1. Code Structure: rather than a series of nebulous scripts, I've architected a modular, extensible, object-oriented solution. For example, the class NeuralNetFactory (oag_int/objects/NeuralNetFactory.py) instantiates various neural nets which can then be trained and reused throughout the code. The code takes as input parameters (see oag_int/config/runparams.py) which are written to files before training runs to record conditions of past trials (neural net parameters, which columns to train in, etc) for iterative experimentation. Hopefully the structure is simple and intuitive enough to parse, but I can of course offer clarification as needed.
   
2. ML: For this exercise I experimented with a number of modifications to a simple neural network designed for general multivariable function approximation (oag_int/neural_net/model_building.py). The basic neural net consists of a single hidden layer with tan-sigmoid activations to induce nonlinearity, output through a single neuron linear output layer.


Bonus (oag_int/main_test.py): To loosely verify the function of the neural nets that I built for this exercise, I train and evaluate them on a noisy, multivariable polynomial with adjustable gaussian noise. The RMSE residuals reliably converge to the width of the noise function, suggesting some degree of validity.

Key points:
Modularity
Experimental Reproducability

# Data Parsing

# ML
### Neural Network Architectures and Factory

