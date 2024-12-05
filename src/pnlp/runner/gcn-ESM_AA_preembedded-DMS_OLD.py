#!/usr/bin/env python
"""
Model runner for GCN model (single target).
This one loads in from parquet of preembedded ESM AA sequences. 
"""
import os
import tqdm
import time
import torch
import random
import datetime
import numpy as np
from typing import Union
from torch import nn
from torch.utils.data import DataLoader

from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from runner_util_dms import (
    DMSEmbeddedDataset,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
)

class GraphSAGE(nn.Module):
    """ GraphSAGE. """

    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, output_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

# MODEL RUNNING
def run_model(model, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    loss_fn = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")

    starting_epoch = 1
    best_rmse = float('inf')

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
        else: fa.write(f"Epoch,Train MSE,Train RMSE,Test MSE,Test RMSE\n")

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_mse, train_rmse = epoch_iteration(model, loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_mse, test_rmse = epoch_iteration(model, loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train RMSE Loss: {train_rmse:.4f}, Test RMSE Loss: {test_rmse:.4f}')  

        with open(metrics_csv, "a") as fa:        
            fa.write(f"{epoch},{train_mse},{train_rmse},{test_mse},{test_rmse}\n")
            fa.flush()

        # Save best
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            model_path = os.path.join(run_dir, f'best_saved_model.pth')
            print(f"NEW BEST model: RMSE loss {best_rmse:.4f}")
            save_model(model, optimizer, model_path, epoch, test_rmse)
        
        # Save every 100 epochs
        if epoch > 0 and epoch % 100 == 0:
            model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
            save_model(model, optimizer, model_path, epoch, test_rmse)

        # Save checkpoint 
        model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
        save_model(model, optimizer, model_path, epoch, test_rmse)
            
        print("")
        
    plot_log_file(metrics_csv, metrics_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, loss_fn, optimizer, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')

    total_loss = 0
    total_items = 0

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, embeddings, targets = batch_data
        embeddings, targets = embeddings.to(device), targets.to(device).float()

        # Graph Construction
        graphs = []
        for embedding, target in zip(embeddings, targets):
            edges = [(i, i+1) for i in range(embedding.size(0) - 1)]
            edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

            graphs.append(Data(
                x=embedding, 
                edge_index=edge_index,
                y = target.view(-1, 1)
            ))
        batch_graph = Batch.from_data_list(graphs).to(device)
   
        if mode == 'train':
            optimizer.zero_grad()
            preds = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
            batch_loss = loss_fn(preds, batch_graph.y)
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                preds = model(batch_graph.x, batch_graph.edge_index, batch_graph.batch)
                batch_loss = loss_fn(preds, batch_graph.y)

        total_loss += batch_loss.item()
        total_items += targets.size(0)
    
    # total loss is the sum of squared errors over items encountered
    # so divide by the number of items encountered
    # we get mse and rmse per item
    mse = total_loss/total_items
    rmse = np.sqrt(mse)

    return mse, rmse 

if __name__=='__main__':

    # Run setup
    n_epochs = 1000
    batch_size = 64
    max_batch = -1
    num_workers = 4
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data/results directories
    result_tag = 'binding' # specify expression or binding
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/gcn-ESM_AA_preembedded')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"adam.lr{lr}.gcn-ESM_AA_preembedded-DMS_OLD-{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = DMSEmbeddedDataset(os.path.join(data_dir, "pt/mutation_combined_DMS_OLD_train_ESM-AA-embedded.pt"), result_tag)
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

    test_dataset = DMSEmbeddedDataset(os.path.join(data_dir, "pt/mutation_combined_DMS_OLD_test_ESM-AA-embedded.pt"), result_tag)
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

    # GraphSAGE input
    size = 320
    input_channels = size # Number of input channels (dimensions of the embeddings)
    hidden_channels = size
    out_channels = 1  # For regression output
    model = GraphSAGE(input_channels, hidden_channels, out_channels)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"gcn-ESM_AA_preembedded-DMS_OLD_{result_tag}-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)