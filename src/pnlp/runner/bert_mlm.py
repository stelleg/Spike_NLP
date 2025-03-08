#!/usr/bin/env python
"""
Model runner for BERT-MLM model (NOT ESM-initialized).
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
import lightning as L
from torch.utils.data import DataLoader
from collections import defaultdict

from pnlp.model.language import BERT, ProteinLM
from pnlp.embedding.tokenizer import ProteinTokenizer, token_to_index

from runner_util_rbd_bert_mlm import (
    RBDDataset,
    ScheduledOptim,
    count_parameters,
    save_model,
    load_model,
    load_model_checkpoint,
    plot_log_file,
    plot_aa_preds_heatmap
)

if __name__=='__main__':
    # needed for tensor core support (at least for amd) 
    torch.set_float32_matmul_precision('medium')

    # Run setup
    n_epochs = 20
    batch_size = 256
    num_workers = 4
    lr = 1e-5

    data_dir = os.path.join(os.path.dirname(__file__), f'../../../data/rbd')

    # Create Dataset and DataLoader
    torch.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_train.csv"))
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=False, 
        num_workers=num_workers, 
        worker_init_fn=seed_worker, 
        generator=torch.Generator().manual_seed(0)
    )

    test_dataset = RBDDataset(os.path.join(data_dir, "spikeprot0528.clean.uniq.noX.RBD.metadata.variants_test.csv"))
    test_data_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
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
    tokenizer = ProteinTokenizer(max_len, mask_prob)

    model = ProteinLM(bert, tokenizer, vocab_size=len(token_to_index))

    # train
    trainer = L.Trainer(
        max_epochs=n_epochs,
        num_nodes=4,
        strategy='deepspeed')
    trainer.fit(model=model, train_dataloaders=train_data_loader)

    # model is checkpointed in ./lightning_logs
    # todo: load checkpoint

    # test
    trainer.test(model=model, dataloaders=test_data_loader)

