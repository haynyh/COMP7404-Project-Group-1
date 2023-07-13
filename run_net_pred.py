from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


from network import my_softplus, my_softplus_derivative, Net
from train_utils import train, test
from data_utils import get_data_loaders, task_list, set_task, save_pklgz, get_train_np_loader, load_chem_data, get_pred_data_loaders
from utils import print_args, AverageMeter
from arg_utils import get_args

import numpy as np
import os
from time import time
import argparse
from sklearn.metrics import roc_auc_score
import pandas as pd


def predict(args, model, device, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            #output = model(data)
            output, relu_masks = model(data, p=0, training=False)
            #print(output)
            output = torch.softmax(output, dim=-1)
            pred = output.max(1, keepdim=True)[1]
            #_, pred = torch.max(output, 1)
            predictions = pred

    return predictions

def main():
    # from some github repo...
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = get_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available() 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net(input_dim=args.input_dim, output_dim=args.output_dim, hidden_dim=args.hidden_dim, num_layer=args.depth, num_back_layer=args.back_n, dense=True, drop_type=args.drop_type, net_type=args.net_type, approx=args.anneal).to(device)
 
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'AMSGrad':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.gamma)
    
    if args.anneal == 'approx':
        args.net_type = 'approx_' + args.net_type 

    best_model_name = './checkpoint/{}/{}/best_seed{}_depth{}_ckpt.t7'.format(args.dataset.strip('/'), args.net_type, args.seed, args.depth)
    checkpoint = torch.load(best_model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # Load your new dataset
    # Make sure your new dataset is formatted properly
    # You may need to create a custom DataLoader for your new dataset.
    #data_df = './prediction/valid.fgp2048.csv'
    
    #show original dataset for prediction
    dir_name = 'prediction/' + args.dataset
    df_pred = pd.read_csv(dir_name + 'pred.fgp{}.csv'.format(args.input_dim))
    print(" ")
    print("Data before prediction")
    print(" ")
    print(df_pred)
    
    pred_loader = get_pred_data_loaders(args.dataset, args.batch_size, sub_task=args.sub_task, dim=args.input_dim)
    predictions = predict(args, model, device, pred_loader)
    predictions_list = [p[0] for p in predictions.tolist()]
    df_pred['Class'] = df_pred['Class'].astype(str).apply(lambda x: x.strip())
    df_pred['Class'] = predictions_list
    print("Data after prediction")
    print(" ")
    print(df_pred)
    
    df_pred.to_csv(dir_name + 'pred.fgp{}.csv'.format(args.input_dim), index=False)
    
    #x_np = data_df.iloc[0:].values.astype(np.float32)
    #x_np = data_df.iloc[:, 1].values.astype(np.float32)
    #print(x_np)
    #x_tensor = torch.tensor(x_np).to(device)
    #batch_size = x_tensor.shape[0]
    #x_tensor = x_tensor.reshape(batch_size, -1, 1)

   
    
    

if __name__ == '__main__':
    main()

