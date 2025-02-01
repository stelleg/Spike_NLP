#!/usr/bin/env python
"""
Model runner for BERT-GCN model (single target).
BERT weights initialized with finetuned BERT_MLM-ESM_INIT weights.
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

from pnlp.model.language import BERT, ProteinMaskedLanguageModel
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index

from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from runner_util_dms_bert_mlm import (
    DMSDataset,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
)

class GraphSAGE(nn.Module):
    def __init__(self, input_channels, hidden_channels, fcn_num_layers):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # FCN layer(s)
        layers = []

        for _ in range(fcn_num_layers):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())

        self.fcn = nn.Sequential(*layers)
        
        self.output = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fcn(x)
        output = self.output(x).squeeze(1)
        return output

class BERT_GCN(nn.Module):
    def __init__(self, bert, gcn, vocab_size):
        super().__init__()
        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)
        self.gcn = gcn

    def forward(self, x, targets):
        with torch.set_grad_enabled(self.training):  # Enable gradients, managed by model.eval() or model.train() in epoch_iteration
            bert_embedding = self.bert(x)
            error_1 = self.mlm(bert_embedding) # error from masked language
        
            # Graph Construction
            graphs = []
            for embedding, target in zip(bert_embedding, targets):
                edges = [(i, i+1) for i in range(embedding.size(0) - 1)]
                edge_index = torch.tensor(edges, dtype=torch.int64).t().contiguous()

                graphs.append(Data(
                    x=embedding, 
                    edge_index=edge_index,
                    y=torch.tensor([target], dtype=torch.float32)
                ))

            batch_graph = Batch.from_data_list(graphs).to(device)
            error_2 = self.gcn(batch_graph.x, batch_graph.edge_index, batch_graph.batch) # error from regession

        return error_1, error_2, batch_graph.y

# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    gcn_loss_fn = nn.MSELoss(reduction='sum').to(device) 
    mlm_loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics_csv = os.path.join(run_dir, f"{save_as}_metrics.csv")
    metrics_img = os.path.join(run_dir, f"{save_as}_metrics.pdf")

    starting_epoch = 1
    best_mlm_accuracy = 0
    best_loss = float('inf')
    
    # Load saved model
    if saved_model_pth is not None and os.path.exists(saved_model_pth):
        if from_checkpoint:
            model_state, optimizer_state, starting_epoch, best_mlm_accuracy, best_loss = load_model(saved_model_pth, device)

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
            starting_epoch += 1
            
            if starting_epoch > n_epochs:
                raise ValueError(f"Starting epoch ({starting_epoch}) is greater than the total number of epochs to run ({n_epochs}). Adjust the number of epochs, 'n_epochs'.")
        
        else:
            model_state, _, _, _, _ = load_model(saved_model_pth, device)
            model.load_state_dict(model_state)

    with open(metrics_csv, "a") as fa:
        if from_checkpoint: load_model_checkpoint(saved_model_pth, metrics_csv, starting_epoch)
        else: 
            fa.write(
                f"Epoch,"
                f"Train MLM Accuracy,Test MLM Accuracy,"
                f"Train MLM Loss,Test MLM Loss,"
                f"Train GCN MSE,Test GCN MSE,"
                f"Train GCN RMSE,Test GCN RMSE,"
                f"Train Loss,Test Loss\n"
            )

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_mlm_accuracy, train_mlm_loss, train_gcn_mse, train_gcn_rmse, train_loss = epoch_iteration(model, tokenizer, mlm_loss_fn, gcn_loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_mlm_accuracy, test_mlm_loss, test_gcn_mse, test_gcn_rmse, test_loss = epoch_iteration(model, tokenizer, mlm_loss_fn, gcn_loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train MLM Accuracy: {train_mlm_accuracy:.4f}, Test MLM Accuracy: {test_mlm_accuracy:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train MLM Loss: {train_mlm_loss:.4f}, Test MLM Loss: {test_mlm_loss:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train GCN RMSE: {train_gcn_rmse:.4f}, Test GCN RMSE: {test_gcn_rmse:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n')
        
        with open(metrics_csv, "a") as fa:  
            fa.write(
                f"{epoch},"
                f"{train_mlm_accuracy},{test_mlm_accuracy},"
                f"{train_mlm_loss},{test_mlm_loss},"
                f"{train_gcn_mse},{test_gcn_mse},"
                f"{train_gcn_rmse},{test_gcn_rmse},"
                f"{train_loss},{test_loss}\n"
            )
            fa.flush()

        # Save best
        if test_mlm_accuracy > best_mlm_accuracy or (test_mlm_accuracy == best_mlm_accuracy and test_loss < best_loss):
            best_mlm_accuracy = test_mlm_accuracy
            best_loss = test_loss
            model_path = os.path.join(run_dir, f'best_saved_model.pth')
            print(f"NEW BEST model: accuracy {best_mlm_accuracy:.4f} and loss {best_loss:.4f}")
            save_model(model, optimizer, model_path, epoch, test_mlm_accuracy, test_loss)
        
        # Save every 100 epochs
        if epoch > 0 and epoch % 100 == 0:
            model_path = os.path.join(run_dir, f'saved_model-epoch_{epoch}.pth')
            save_model(model, optimizer, model_path, epoch, test_mlm_accuracy, test_loss)

        # Save checkpoint 
        model_path = os.path.join(run_dir, f'checkpoint_saved_model.pth')
        save_model(model, optimizer, model_path, epoch, test_mlm_accuracy, test_loss)
            
        print("")

    plot_log_file(metrics_csv, metrics_img, "GCN")

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, tokenizer, mlm_loss_fn, gcn_loss_fn, optimizer, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')
    
    total_mlm_loss = 0
    total_gcn_loss = 0
    total_loss = 0
    total_masked = 0
    total_items = 0
    correct_predictions = 0

    # Set max_batch if None
    if not max_batch:
        max_batch = len(data_loader)

    for batch, batch_data in data_iter:
        if max_batch > 0 and batch >= max_batch:
            break

        seq_ids, seqs, targets = batch_data
        targets = targets.to(device).float()
        masked_tokenized_seqs = tokenizer(seqs).to(device) 
        unmasked_tokenized_seqs = tokenizer._batch_pad(seqs).to(device)
   
        if mode == 'train':
            optimizer.zero_grad()
            mlm_preds, gcn_preds, y = model(masked_tokenized_seqs, targets)
            batch_mlm_loss = mlm_loss_fn(mlm_preds.transpose(1, 2), unmasked_tokenized_seqs)
            batch_gcn_loss = gcn_loss_fn(gcn_preds, y)
            batch_loss = (batch_mlm_loss * 0.1) + batch_gcn_loss 
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                mlm_preds, gcn_preds, y = model(masked_tokenized_seqs, targets)
                batch_mlm_loss = mlm_loss_fn(mlm_preds.transpose(1, 2), unmasked_tokenized_seqs)
                batch_gcn_loss = gcn_loss_fn(gcn_preds, y)
                batch_loss = (batch_mlm_loss * 0.1) + batch_gcn_loss

        # Loss
        total_mlm_loss += batch_mlm_loss.item()
        total_gcn_loss += batch_gcn_loss.item()
        total_loss += batch_loss.item()
        total_items += targets.size(0)

        # Accuracy
        predicted_tokens = torch.max(mlm_preds, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions += torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked += masked_locations[0].numel()     

    # Average accuracy/loss per masked token - MLM
    avg_mlm_loss = total_mlm_loss / total_masked
    avg_mlm_accuracy = (correct_predictions / total_masked) * 100

    # RMSE - GCN
    gcn_mse = total_gcn_loss/total_items
    gcn_rmse = np.sqrt(gcn_mse)

    # MLM + GCN Average loss per item
    avg_loss = total_loss / total_items

    return avg_mlm_accuracy, avg_mlm_loss, gcn_mse, gcn_rmse, avg_loss

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
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/bert_gcn')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"5_relu-adam.lr{lr}.bert_gcn-DMS_OLD-{result_tag}-{date_hour_minute}")
    os.makedirs(run_dir, exist_ok = True)

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = DMSDataset(os.path.join(data_dir, "mutation_combined_DMS_OLD_train.csv"), result_tag)
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

    test_dataset = DMSDataset(os.path.join(data_dir, "mutation_combined_DMS_OLD_test.csv"), result_tag)
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
    
    # BERT input
    max_len = 280
    mask_prob = 0.15
    embedding_dim = 320 
    dropout = 0.1
    n_transformer_layers = 12
    n_attn_heads = 10
    bert = BERT(embedding_dim, dropout, max_len, mask_prob, n_transformer_layers, n_attn_heads)
    bert.embedding.load_pretrained_embeddings(os.path.join(data_dir, '../rbd/esm_weights-embedding_dim320.pth'), no_grad=False)
    tokenizer = ProteinTokenizer(max_len, mask_prob)

    # GraphSAGE input
    size = 320
    input_channels = size # Number of input channels (dimensions of the embeddings)
    hidden_channels = size
    fcn_num_layers = 5
    gcn = GraphSAGE(input_channels, hidden_channels, fcn_num_layers)

    # BERT-GCN input
    bert_model_pth = os.path.join(results_dir, "../bert_mlm-esm_init/adam.lr1e-05.bert_mlm-esm_init-RBD-2024-12-04_14-33/best_saved_model.pth")
    saved_state = torch.load(bert_model_pth, map_location=device, weights_only=False)
    model_state = saved_state['model_state_dict']
    bert_state_dict = {key[len('bert.'):]: value for key, value in model_state.items() if key.startswith('bert.')}
    mlm_state_dict = {key[len('mlm.'):]: value for key, value in model_state.items() if key.startswith('mlm.')}

    bert.load_state_dict(bert_state_dict)
    model = BERT_GCN(bert, gcn, len(token_to_index))
    model.mlm.load_state_dict(mlm_state_dict)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"bert_gcn-DMS_OLD_{result_tag}-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)