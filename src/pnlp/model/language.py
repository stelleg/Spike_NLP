"""Protien Language Models"""

import torch
import torch.nn as nn
import lightning as L

from pnlp.model.bert import BERT


class ProteinMaskedLanguageModel(L.LightningModule):
    """Masked language model for protein sequences"""

    def __init__(self, hidden: int, vocab_size: int):
        """
        hidden: input size of the hidden linear layers
        vocab_size: vocabulary size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.linear(x)

# %%
class ProteinLM(L.LightningModule):
    """
    BERT protein language model
    """

    def __init__(self, bert: BERT, tokenizer, vocab_size: int):
        super().__init__()
        self.bert = bert
        self.tokenizer = tokenizer
        self.mlm = ProteinMaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x):
        x = self.bert(x).to(self.device)
        return self.mlm(x)

    def training_step(self, batch, batch_idx):
        _, seqs = batch
        seqs = seqs
        masked_tokenized_seqs = self.tokenizer(seqs).to(self.device)
        unmasked_tokenized_seqs = self.tokenizer._batch_pad(seqs).to(self.device)
        preds = self.forward(masked_tokenized_seqs).to(self.device)

        # loss
        loss = nn.CrossEntropyLoss(reduction='sum')(preds.transpose(1,2), unmasked_tokenized_seqs)
        self.log('loss', loss, on_epoch=True, prog_bar=True, batch_size=256, sync_dist=True)

        # accuracy
        # todo: only count masked locations, requires token_to_index
        predicted_tokens = torch.max(preds, dim=-1)[1]
        correct_predictions = torch.eq(predicted_tokens, unmasked_tokenized_seqs).sum().item()
        total = unmasked_tokenized_seqs.numel()     
        self.log('accuracy', correct_predictions/total, on_epoch=True, prog_bar=True, batch_size=256, sync_dist=True)

        return loss 

    def test_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=0.01)
