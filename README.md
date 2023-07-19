# COMP7404-Project-Group-1
Neural network on oblique decision tree: Local constant network <br />
Group Project Study <br />
Modified from https://github.com/guanghelee/iclr20-lcn

## Repo structure

* BACE Dataset Demo.ipynb: contain scripts for reproducing the training and producing prediction
* Model comparison.ipynb: contain scripts for training 3 datasets in different LCN variations
* [data/](data/): datasets used in the paper. Training/validation/testing splits are provided. 
* [prediction/](prediction/): new data to be generated prediction on
* [log/](log/): the logs of the training results will be stored here.
* [checkpoint/](checkpoint/): the checkpoint of the learned model will be stored here. 

## Main files

* [run_net_training.py](run_net_training.py): main python file for running LCNs/ALCNs/LLNs
* [run_elcn_training.py](run_elcn_training.py): main python file for running ELCN
* [run_net_pred.py](run_net_pred.py): main python file for generating prediction on the new data

## Quick start and reproducing training and prediction


* BACE Dataset Demo.ipynb contain scripts for reproducing the training and producing prediction, please following through for your application on your set of data, make sure the data file format is aligned.
* Model comparison.ipynb: contain scripts for training 3 datasets in different LCN variations. If you want to explore different LCN variations on your data, please follow through the codes to call different network on your dataset.
* The results will be stored in the [log/](log/) directory. The last 3 columns record the training, validation, and testing performance, respectively (from left to right). 

## Guidelines on code demo and using the model for your own data


* BACE Dataset Demo.ipynb is the code demo, running each cells from beginning to the end in the notebook goes through (1) studying the training data, (2) training the model using the training data and pre-defined hyper-parameters, (3) visualizing performance of the trainings done in the log file, (4) studying the new data we want to use the trained model on prediction, (5) load the trained weights with the pre-defined hyper-parameters to reinitiate the model on generating prediction on the new dataset we put in ./prediction folder
* To apply to your own data, please (1) split your training dataset in test_split, train_split and valid_split and store into ./data/application_split, make sure they follow 2048x1 dimension, you can reference the dataset format in BACE/tox21/HIV, (2) follow the demo, change data location to ./application_split and tune hyper-parameters for best performance, (3) prepare your new data that you want to generate prediction on and store it in ./prediction/applpication_split/ (4) call prediction script to generate prediction on it, when calling the script you need to set the hyper-parameters the same as your training hyper-parameters in order to reproduce the same model environment


## How to set hyper-parameters

### Switching among LCN, ALCN, and LLN

The default model in [run_net_training.py](run_net_training.py) is LCN. 

* Switching from LCN to ALCN: setting `--anneal` to `approx`.
* Switching from LCN to LLN: setting `--net_type` to `locally_linear`.

### Some suggestions for hyper-parameter tuning

If you would like to apply the codes for other datasets, we suggest to tune the following hyper-parameters. You can see the complete list of hyper-parameters in [arg_utils.py](arg_utils.py). 

* Depth of the network (`--depth`).
* We suggest to use DropConnect (set `--drop_type` to `node_dropconnect`) and mildly tune the dropping probability (e.g., try `--p` in `{0.25, 0.5, 0.75}`).
* You can start trying the model by setting `--back_n` (the depth of the network *g<sub>&phi;</sub>*) to `0`. If it doesn't work, please try to increase it. In our experiments, we found that we need to increase it for regression tasks, and we can simply keep it to `0` for classification tasks. 
* You may want to tune the learning iterations (`--epochs`), learning rate (`--lr`), and optimizer (`--optimizer`) for your tasks. If you change the learning iterations (`--epochs`), you probably should also change the `--lr_step_size` and `--gamma` (see their meaning in the `help` descriptions in [arg_utils.py](arg_utils.py)).
* You may enlarge the `--batch-size` to accelerate training. 

