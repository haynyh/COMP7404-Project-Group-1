# COMP7404-Project-Group-1
Neural network on oblique decision tree: Local constant network
Group Project Study

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


* BACE Dataset Demo.ipynbcontain scripts for reproducing the training and producing prediction, please following through for your application on your set of data, make sure the data file format is aligned.
* Model comparison.ipynb: contain scripts for training 3 datasets in different LCN variations. If you want to explore different LCN variations on your data, please follow through the codes to call different network on your dataset.
* The results will be stored in the [log/](log/) directory. The last 3 columns record the training, validation, and testing performance, respectively (from left to right). 

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

