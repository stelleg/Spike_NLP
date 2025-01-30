#!/usr/bin/env python
"""
Model runner for ESM-FCN model (multi target for both binding and expression).
"""
import os
import tqdm
import torch
import time
import random
import datetime
import numpy as np
from typing import Union
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmModel

from runner_util_dms import (
    DMSDataset_BE,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file_BE,
)

class FCN(nn.Module):
    """ Fully Connected Network """

    def __init__(self,
                 fcn_input_size,    # The number of input features
                 fcn_hidden_size,   # The number of features in hidden layer of FCN.
                 fcn_num_layers):   # The number of fcn layers  
        super().__init__()

        # Creating a list of layers for the FCN
        # Subsequent layers after 1st should be equal to hidden_size for input_size
        layers = []
        input_size = fcn_input_size

        for _ in range(fcn_num_layers):
            layers.append(nn.Linear(input_size, fcn_hidden_size))
            layers.append(nn.ReLU())
            input_size = fcn_hidden_size

        # FCN layers
        self.fcn = nn.Sequential(*layers)

        # FCN output layers - two separate heads for binding and expression
        self.binding_head = nn.Linear(fcn_hidden_size, 1)
        self.expression_head = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        fcn_out = self.fcn(x)

        # Task-specific predictions
        binding_pred = self.binding_head(fcn_out).squeeze(1) # [batch_size]
        expression_pred = self.expression_head(fcn_out).squeeze(1) # [batch_size]

        return binding_pred, expression_pred

# ESM-FCN
class ESM_FCN(nn.Module):
    def __init__(self, esm, fcn):
        super().__init__()
        self.esm = esm
        self.fcn = fcn

    def forward(self, tokenized_seqs):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            esm_last_hidden_state = self.esm(**tokenized_seqs).last_hidden_state # shape: [batch_size, sequence_length, embedding_dim]
            esm_cls_embedding = esm_last_hidden_state[:, 0, :]  # CLS token embedding (sequence-level representations), [batch_size, embedding_dim]
            binding_preds, expression_preds = self.fcn(esm_cls_embedding) # [batch_size]
        return binding_preds, expression_preds

# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")

    starting_epoch = 1
    best_rmse = float('inf')
    best_binding_rmse = float('inf')
    best_expression_rmse = float('inf')

    # Load saved model
    if saved_model_pth is not None and os.path.exists(saved_model_pth):
        if from_checkpoint:
            model_state, optimizer_state, starting_epoch, best_rmse = load_model(saved_model_pth, device)

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            starting_epoch += 1
            
            if starting_epoch > n_epochs:
                raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
        
        else:
            model_state, _, _, _ = load_model(saved_model_pth, device)
            model.load_state_dict(model_state)

    with open(metrics_csv, "a") as fa:
        if from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else: 
            fa.write((
                "Epoch,"
                "Train Binding MSE,Train Binding RMSE,Train Expression MSE,Train Expression RMSE,Train BE MSE,Train BE RMSE,"
                "Test Binding MSE,Test Binding RMSE,Test Expression MSE,Test Expression RMSE,Test BE MSE,Test BE RMSE\n"
            ))

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_binding_mse, train_binding_rmse, train_expression_mse, train_expression_rmse, train_be_mse, train_be_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_binding_mse, test_binding_rmse, test_expression_mse, test_expression_rmse, test_be_mse, test_be_rmse = epoch_iteration(model, tokenizer, loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train Binding RMSE: {train_binding_rmse:.4f}, Train Expression RMSE: {train_expression_rmse:.4f}, Train BE RMSE: {train_be_rmse:.4f}') 
        print(f'{" "*(8+len(str(epoch)))} Test Binding RMSE: {test_binding_rmse:.4f}, Test Expression RMSE: {test_expression_rmse:.4f}, Test BE RMSE: {test_be_rmse:.4f}') 

        with open(metrics_csv, "a") as fa:         
            fa.write((
                f"{epoch}," 
                f"{train_binding_mse},{train_binding_rmse},{train_expression_mse},{train_expression_rmse},{train_be_mse},{train_be_rmse},"
                f"{test_binding_mse},{test_binding_rmse},{test_expression_mse},{test_expression_rmse},{test_be_mse},{test_be_rmse}\n"
            ))                
            fa.flush()

        # Save best
        if test_be_rmse < best_rmse:
            best_rmse = test_be_rmse
            model_path = os.path.join(run_dir, f'best_saved_model.pth')
            print(f"NEW BEST model: RMSE BE loss {best_rmse:.4f}")
            save_model(model, optimizer, model_path, epoch, test_be_rmse)

        if test_binding_rmse < best_binding_rmse:
            best_binding_rmse = test_binding_rmse
            model_path = os.path.join(run_dir, f'best_saved_binding_model.pth')
            print(f"NEW BEST binding model: RMSE binding loss {best_binding_rmse:.4f}")
            save_model(model, optimizer, model_path, epoch, test_binding_rmse)

        if test_expression_rmse < best_expression_rmse:
            best_expression_rmse = test_expression_rmse
            model_path = os.path.join(run_dir, f'best_saved_expression_model.pth')
            print(f"NEW BEST expression model: RMSE expression loss {best_expression_rmse:.4f}")
            save_model(model, optimizer, model_path, epoch, test_expression_rmse)
        
        # Save every 100 epochs
        if epoch > 0 and epoch % 100 == 0:
            model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
            save_model(model, optimizer, model_path, epoch, test_be_rmse)

        # Save checkpoint 
        model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
        save_model(model, optimizer, model_path, epoch, test_be_rmse)
            
        print("")
        
    plot_log_file_BE(metrics_csv, metrics_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, tokenizer, loss_fn, optimizer, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_binding_loss = 0
    total_expression_loss = 0
    total_be_loss = 0
    total_items = 0

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, seqs, binding_targets, expression_targets = batch_data
        binding_targets, expression_targets = binding_targets.to(device).float(), expression_targets.to(device).float()
        tokenized_seqs = tokenizer(seqs, return_tensors="pt").to(device)
   
        if mode == 'train':
            optimizer.zero_grad()
            binding_preds, expression_preds = model(tokenized_seqs)
            binding_loss = loss_fn(binding_preds, binding_targets)
            expression_loss = loss_fn(expression_preds, expression_targets)
            batch_be_loss = binding_loss + expression_loss
            batch_be_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                binding_preds, expression_preds = model(tokenized_seqs)
                binding_loss = loss_fn(binding_preds, binding_targets)
                expression_loss = loss_fn(expression_preds, expression_targets)
                batch_be_loss = binding_loss + expression_loss

        total_binding_loss += binding_loss.item()
        total_expression_loss += expression_loss.item()
        total_be_loss += batch_be_loss.item()
        total_items += binding_targets.size(0)  # same size as expression_targets.size(0)
    
    # total loss is the sum of squared errors over items encountered
    # so divide by the number of items encountered
    # we get mse and rmse per item
    binding_mse = total_binding_loss/total_items
    expression_mse = total_expression_loss/total_items
    be_mse = total_be_loss/total_items

    binding_rmse = np.sqrt(binding_mse)
    expression_rmse = np.sqrt(expression_mse)
    be_rmse = np.sqrt(be_mse)

    return binding_mse, binding_rmse, expression_mse, expression_rmse, be_mse, be_rmse

if __name__=='__main__':

    # Run setup
    n_epochs = 100
    batch_size = 64
    max_batch = -1
    num_workers = 4
    lr = 1e-4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/esm_fcn')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"binding_test-adam.lr{lr}.esm_fcn_BE-DMS_OLD-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = DMSDataset_BE(os.path.join(data_dir, "mutation_combined_DMS_OLD_train.csv"))
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker, 
        generator=torch.Generator().manual_seed(0), 
        pin_memory=True
    )

    test_dataset = DMSDataset_BE(os.path.join(data_dir, "mutation_combined_DMS_OLD_test.csv"))
    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker, 
        generator=torch.Generator().manual_seed(0), 
        pin_memory=True
    )

    # ESM input
    esm_version = "facebook/esm2_t6_8M_UR50D" 
    esm = EsmModel.from_pretrained(esm_version, cache_dir='../../../../model_downloads').to(device)
    tokenizer = AutoTokenizer.from_pretrained(esm_version, cache_dir='../../../../model_downloads')

    # FCN input
    size = 320
    fcn_input_size = size  
    fcn_hidden_size = size
    fcn_num_layers = 5
    fcn = FCN(fcn_input_size, fcn_hidden_size, fcn_num_layers)

    model = ESM_FCN(esm, fcn)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"esm_fcn_BE-DMS_OLD_BE-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)