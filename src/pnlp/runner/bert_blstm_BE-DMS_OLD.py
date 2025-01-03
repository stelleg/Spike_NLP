#!/usr/bin/env python
"""
Model runner for BERT-BLSTM model (multi target for both binding and expression).
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

from runner_util_dms_bert_mlm import (
    DMSDataset_BE,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file_BE,
)

# BLSTM
class BLSTM(nn.Module):
    """ Bidirectional LSTM. Output is embedding layer, not prediction value."""

    def __init__(self,
                 lstm_input_size,    # The number of expected features.
                 lstm_hidden_size,   # The number of features in hidden state h.
                 lstm_num_layers,    # Number of recurrent layers in LSTM.
                 lstm_bidirectional, # Bidrectional LSTM.
                 fcn_hidden_size,    # The number of features in hidden layer of CN.
                 fcn_num_layers):    # The number of fcn layers
        super().__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            bidirectional=lstm_bidirectional,
                            batch_first=True)           

        # FCN layer(s)
        layers = []
        input_size = 2 * lstm_hidden_size if lstm_bidirectional else lstm_hidden_size

        for _ in range(fcn_num_layers):
            layers.append(nn.Linear(input_size, fcn_hidden_size))
            layers.append(nn.ReLU())
            input_size = fcn_hidden_size

        self.fcn = nn.Sequential(*layers)

        # FCN output layers - two separate heads for binding and expression
        self.binding_head = nn.Linear(fcn_hidden_size, 1)
        self.expression_head = nn.Linear(fcn_hidden_size, 1)

    def forward(self, x):
        num_directions = 2 if self.lstm.bidirectional else 1
        h_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c_0 = torch.zeros(num_directions * self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        lstm_out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        lstm_final_out = lstm_out[:, -1, :] 
        fcn_out = self.fcn(lstm_final_out)

        # Task-specific predictions
        binding_pred = self.binding_head(fcn_out).squeeze(1) # [batch_size]
        expression_pred = self.expression_head(fcn_out).squeeze(1) # [batch_size]

        return binding_pred, expression_pred

# BERT-BLSTM
class BERT_BLSTM(nn.Module):
    def __init__(self, bert, blstm, vocab_size):
        super().__init__()
        self.bert = bert
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)
        self.blstm = blstm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bert(x)
        error_1 = self.mlm(x) # error from masked language
        error_2_binding, error_2_expression = self.blstm(x) # error from regession
        return error_1, error_2_binding, error_2_expression
    
# MODEL RUNNING
def run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs: int, lr:float, max_batch: Union[int, None], device: str, run_dir: str, save_as: str, saved_model_pth:str=None, from_checkpoint:bool=False):
    """ Run a model through train and test epochs. """

    model = model.to(device)
    regression_loss_fn = nn.MSELoss(reduction='sum').to(device) 
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
                f"Train Binding MSE,Test Binding MSE,"
                f"Train Binding RMSE,Test Binding RMSE,"
                f"Train Expression MSE,Test Expression MSE,"
                f"Train Expression RMSE,Test Expression RMSE,"
                f"Train BE MSE,Test BE MSE,"
                f"Train BE RMSE,Test BE RMSE,"
                f"Train Loss,Test Loss\n"
            )

    # Running
    start_time = time.time()

    for epoch in range(starting_epoch, n_epochs + 1):
        train_mlm_accuracy, train_mlm_loss, train_binding_mse, train_binding_rmse, train_expression_mse, train_expression_rmse, train_be_mse, train_be_rmse, train_loss = epoch_iteration(model, tokenizer, mlm_loss_fn, regression_loss_fn, optimizer, train_data_loader, epoch, max_batch, device, mode='train')
        test_mlm_accuracy, test_mlm_loss, test_binding_mse, test_binding_rmse, test_expression_mse, test_expression_rmse, test_be_mse, test_be_rmse, test_loss = epoch_iteration(model, tokenizer, mlm_loss_fn, regression_loss_fn, optimizer, test_data_loader, epoch, max_batch, device, mode='test')

        print(f'Epoch {epoch} | Train MLM Accuracy: {train_mlm_accuracy:.4f}, Test MLM Accuracy: {test_mlm_accuracy:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train MLM Loss: {train_mlm_loss:.4f}, Test MLM Loss: {test_mlm_loss:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train Binding RMSE: {train_binding_rmse:.4f}, Test Binding RMSE: {test_binding_rmse:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train Expression RMSE: {train_expression_rmse:.4f}, Test Expression RMSE: {test_expression_rmse:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train BE RMSE: {train_be_rmse:.4f}, Test BE RMSE: {test_be_rmse:.4f}')
        print(f'{" "*(8+len(str(epoch)))} Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}\n')
        
        with open(metrics_csv, "a") as fa:  
            fa.write(
                f"{epoch},"
                f"{train_mlm_accuracy},{test_mlm_accuracy},"
                f"{train_mlm_loss},{test_mlm_loss},"
                f"{train_binding_mse},{test_binding_mse},"
                f"{train_binding_rmse},{test_binding_rmse},"
                f"{train_expression_mse},{test_expression_mse},"
                f"{train_expression_rmse},{test_expression_rmse},"
                f"{train_be_mse},{test_be_mse},"
                f"{train_be_rmse},{test_be_rmse},"
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

    plot_log_file_BE(metrics_csv, metrics_img)

    # End timer and print duration
    end_time = time.time()
    duration = end_time - start_time
    formatted_duration = str(datetime.timedelta(seconds=duration))
    print(f'Training and testing complete in: {formatted_duration} (D day(s), H:MM:SS.microseconds)')

def epoch_iteration(model, tokenizer, mlm_loss_fn, regression_loss_fn, optimizer, data_loader, epoch, max_batch, device, mode):
    """ Used in run_model. """
    
    model.train() if mode=='train' else model.eval()

    data_iter = tqdm.tqdm(enumerate(data_loader),
                          desc=f'Epoch_{mode}: {epoch}',
                          total=len(data_loader),
                          bar_format='{l_bar}{r_bar}')
    
    total_mlm_loss = 0
    total_binding_loss = 0
    total_expression_loss = 0
    total_be_loss = 0
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

        seq_ids, seqs, binding_targets, expression_targets = batch_data
        binding_targets, expression_targets = binding_targets.to(device).float(), expression_targets.to(device).float()
        masked_tokenized_seqs = tokenizer(seqs).to(device) 
        unmasked_tokenized_seqs = tokenizer._batch_pad(seqs).to(device)
   
        if mode == 'train':
            optimizer.zero_grad()
            mlm_preds, binding_preds, expression_preds = model(masked_tokenized_seqs)
            batch_mlm_loss = mlm_loss_fn(mlm_preds.transpose(1, 2), unmasked_tokenized_seqs)
            batch_binding_loss = regression_loss_fn(binding_preds, binding_targets)
            batch_expression_loss = regression_loss_fn(expression_preds, expression_targets)
            batch_be_loss = batch_binding_loss + batch_expression_loss
            batch_loss = (batch_mlm_loss * 0.1) + batch_be_loss 
            batch_loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                mlm_preds, binding_preds, expression_preds = model(masked_tokenized_seqs)
                batch_mlm_loss = mlm_loss_fn(mlm_preds.transpose(1, 2), unmasked_tokenized_seqs)
                batch_binding_loss = regression_loss_fn(binding_preds, binding_targets)
                batch_expression_loss = regression_loss_fn(expression_preds, expression_targets)
                batch_be_loss = batch_binding_loss + batch_expression_loss
                batch_loss = (batch_mlm_loss * 0.1) + batch_be_loss 

        # Loss
        total_mlm_loss += batch_mlm_loss.item()
        total_binding_loss += batch_binding_loss.item()
        total_expression_loss += batch_expression_loss.item()
        total_be_loss +=  batch_be_loss.item()
        total_loss += batch_loss.item()
        total_items += binding_targets.size(0)

        # Accuracy
        predicted_tokens = torch.max(mlm_preds, dim=-1)[1]
        masked_locations = torch.nonzero(torch.eq(masked_tokenized_seqs, token_to_index['<MASK>']), as_tuple=True)
        correct_predictions += torch.eq(predicted_tokens[masked_locations], unmasked_tokenized_seqs[masked_locations]).sum().item()
        total_masked += masked_locations[0].numel()     

    # Average accuracy/loss per masked token - MLM
    avg_mlm_loss = total_mlm_loss / total_masked
    avg_mlm_accuracy = (correct_predictions / total_masked) * 100

    # RMSE - Regression
    binding_mse = total_binding_loss/total_items
    expression_mse = total_expression_loss/total_items
    be_mse = total_be_loss/total_items

    binding_rmse = np.sqrt(binding_mse)
    expression_rmse = np.sqrt(expression_mse)
    be_rmse = np.sqrt(be_mse)

    # MLM + regression average loss per item
    avg_loss = total_loss / total_items

    return avg_mlm_accuracy, avg_mlm_loss, binding_mse, binding_rmse, expression_mse, expression_rmse, be_mse, be_rmse, avg_loss
      
if __name__=='__main__':

    # Run setup
    n_epochs = 1000
    batch_size = 64
    max_batch = -1
    num_workers = 4
    lr = 1e-5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data/results directories
    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/dms') 
    results_dir = os.path.join(os.path.dirname(__file__), f'../../../results/run_results/bert_blstm')

    # Create run directory for results
    now = datetime.datetime.now()
    date_hour_minute = now.strftime("%Y-%m-%d_%H-%M")
    run_dir = os.path.join(results_dir, f"adam.lr{lr}.bert_blstm_BE-DMS_OLD-{date_hour_minute}")
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

    # BLSTM input
    size = 320
    lstm_input_size = size
    lstm_hidden_size = size
    lstm_num_layers = 1        
    lstm_bidrectional = True   
    fcn_hidden_size = size
    fcn_num_layers = 5
    blstm = BLSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, lstm_bidrectional, fcn_hidden_size, fcn_num_layers)

    # BERT-BLSTM input
    bert_model_pth = os.path.join(results_dir, "../bert_mlm-esm_init/adam.lr1e-05.bert_mlm-esm_init-RBD-2024-12-04_14-33/best_saved_model.pth")
    saved_state = torch.load(bert_model_pth, map_location=device, weights_only=False)
    model_state = saved_state['model_state_dict']
    bert_state_dict = {key[len('bert.'):]: value for key, value in model_state.items() if key.startswith('bert.')}
    mlm_state_dict = {key[len('mlm.'):]: value for key, value in model_state.items() if key.startswith('mlm.')}

    bert.load_state_dict(bert_state_dict)
    model = BERT_BLSTM(bert, blstm, len(token_to_index))
    model.mlm.load_state_dict(mlm_state_dict)

    # Run
    count_parameters(model)
    saved_model_pth = None
    from_checkpoint = False
    save_as = f"bert_blstm_BE-DMS_OLD-train_{len(train_dataset)}_test_{len(test_dataset)}"
    run_model(model, tokenizer, train_data_loader, test_data_loader, n_epochs, lr, max_batch, device, run_dir, save_as, saved_model_pth, from_checkpoint)