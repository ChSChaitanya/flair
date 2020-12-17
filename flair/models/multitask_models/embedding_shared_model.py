import logging
import sys

from typing import Dict, List, Union

import torch
import torch.nn as nn

import flair.nn
from flair.data import Dictionary, Sentence
from flair.embeddings import TokenEmbeddings

from .utils import *
from .chaincrf import ChainCRF

log = logging.getLogger("flair")

START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"

class EmbeddingSharedModel(flair.nn.Model):
    def __init__(self,
                 hidden_size: int,
                 embeddings: TokenEmbeddings,
                 tag_dictionary: Dictionary,
                 tag_type: str,
                 use_crf: bool = True,
                 rnn_type: str = "LSTM",
                 rnn_layers: int = 1,
                 dropout: float = 0.0,
                 word_dropout: float = 0.05,
                 locked_dropout: float = 0.5,
                 use_lm=True,
                 lm_loss: float = 0.05,
                 lm_mode: str = 'unshared',
                 pickle_module: str = "pickle",
                 loss_weights: Dict[str, float] = None,
                 bigram=True,
                 ):
        super(EmbeddingSharedModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_layers: int = rnn_layers

        self.use_crf = use_crf
        self.use_lm = use_lm
        self.lm_mode = lm_mode
        self.lm_loss = lm_loss

        self.trained_epochs: int = 0

        self.embeddings = embeddings

        #TODO multicorpus
        self.tag_dictionary.add_item(START_TAG)
        self.tag_dictionary.add_item(STOP_TAG)

        self.tag_type: str = tag_type
        self.tagset_size: int = len(tag_dictionary)

        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.tag_dictionary)
            weight_list = [1. for i in range(n_classes)]
            for i, tag in enumerate(self.tag_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # initialize the network architecture
        self.nlayers: int = rnn_layers
        self.hidden_word = None

        # dropouts
        self.use_dropout: float = dropout
        self.use_word_dropout: float = word_dropout
        self.use_locked_dropout: float = locked_dropout

        self.pickle_module = pickle_module

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(dropout)

        if word_dropout > 0.0:
            self.word_dropout = flair.nn.WordDropout(word_dropout)

        if locked_dropout > 0.0:
            self.locked_dropout = flair.nn.LockedDropout(locked_dropout)

        embedding_dim: int = self.embeddings.embedding_length
        rnn_input_dim: int = embedding_dim

        if rnn_type in ["RNN", "LSTM", "GRU"]:

            for task_id, task in enumerate(tasks):
                setattr(self,
                        f"rnn_{task_id}",
                        getattr(torch.nn, self.rnn_type)(
                            rnn_input_dim,
                            hidden_size,
                            num_layers=self.nlayers,
                            dropout=0.0 if self.nlayers == 1 else 0.5,
                            bidirectional=True,
                            batch_first=True,
                        ))

                if self.use_crf:
                    setattr(self,
                            f"crf_{task_id}",
                            #TODO task specific tagset size
                            ChainCRF(hidden_size * 2, self.tagset_size, bigram=bigram))

                else:
                    setattr(self,
                            f"dense_softmax_{task_id}",
                            #TODO task specific tagset size
                            nn.Linear(hidden_size * 2, self.tagset_size))

        else:

            raise AttributeError("Unknown or unsupported rnn_type.")

        if self.use_lm:

            if self.lm_mode == 'unshared':

                self.dense_fw_1 = nn.Linear(hidden_size, num_words)
                self.dense_bw_1 = nn.Linear(hidden_size, num_words)
                self.dense_fw_2 = nn.Linear(hidden_size, num_words)
                self.dense_bw_2 = nn.Linear(hidden_size, num_words)

            elif self.lm_mode == 'shared':
                self.dense_fw = nn.Linear(hidden_size, num_words)
                self.dense_bw = nn.Linear(hidden_size, num_words)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(size_average=False, reduce=False)


    def forward_loss(self, data_points: Union[List[Sentence], Sentence], sort=True) -> torch.tensor:
        features = self.forward(data_points)
        return self._calculate_loss(features, data_points)

    def forward(self, sentences: List[Sentence]):

        self.embeddings.embed(sentences)

        names = self.embeddings.get_names()

        lengths: List[int] = [len(sentence.tokens) for sentence in sentences]
        longest_token_sequence_in_batch: int = max(lengths)

        pre_allocated_zero_tensor = torch.zeros(
            self.embeddings.embedding_length * longest_token_sequence_in_batch,
            dtype=torch.float,
            device=flair.device,
        )

        all_embs = list()
        for sentence in sentences:
            all_embs += [
                emb for token in sentence for emb in token.get_each_embedding(names)
            ]
            nb_padding_tokens = longest_token_sequence_in_batch - len(sentence)

            if nb_padding_tokens > 0:
                t = pre_allocated_zero_tensor[
                    : self.embeddings.embedding_length * nb_padding_tokens
                    ]
                all_embs.append(t)

        sentence_tensor = torch.cat(all_embs).view(
            [
                len(sentences),
                longest_token_sequence_in_batch,
                self.embeddings.embedding_length,
            ]
        )

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        if self.use_word_dropout > 0.0:
            sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        packed = torch.nn.utils.rnn.pack_padded_sequence(
            sentence_tensor, lengths, enforce_sorted=False, batch_first=True
        )

        #TODO task specific rnn
        rnn_output, hidden = self.rnn(packed)

        if self.use_dropout > 0.0:
            sentence_tensor = self.dropout(sentence_tensor)
        # word dropout only before LSTM - TODO: more experimentation needed
        # if self.use_word_dropout > 0.0:
        #     sentence_tensor = self.word_dropout(sentence_tensor)
        if self.use_locked_dropout > 0.0:
            sentence_tensor = self.locked_dropout(sentence_tensor)

        features = self.linear(sentence_tensor)

        return features

        # output from rnn [batch, length, tag_space]
        output, _, mask, length = self._get_rnn_output(input_word, input_char, main_task, mask, hx=hx)
        # [batch, length, num_label,  num_label]
        return self.crf(output, mask=mask), mask

    def loss(self, input_word, input_char, target, main_task, target_fw, target_bw, mask, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        if self.use_lm:
            output, _, mask, length, lm_fw, lm_bw = self._get_rnn_output(input_word, input_char, main_task, mask, hx=hx)
            if self.lm_mode == 'unshared':
                if main_task:
                    lm_fw = self.dense_fw_2(lm_fw)
                    lm_bw = self.dense_bw_2(lm_bw)
                else:
                    lm_fw = self.dense_fw_1(lm_fw)
                    lm_bw = self.dense_bw_1(lm_bw)
            elif self.lm_mode == 'shared':
                lm_fw = self.dense_fw(lm_fw)
                lm_bw = self.dense_bw(lm_bw)
            else:
                raise ValueError('Unknown LM mode: %s' % self.lm_mode)
            output_size = lm_fw.size()
            output_size = (output_size[0] * output_size[1], output_size[2])
            lm_fw = lm_fw.view(output_size)
            lm_bw = lm_bw.view(output_size)
            max_len = length.max()
            target_fw = target_fw[:, :max_len].contiguous()
            target_bw = target_bw[:, :max_len].contiguous()
            fw_loss = (self.nll_loss(self.logsoftmax(lm_fw), target_fw.view(-1)) * mask.contiguous().view(
                -1)).sum() / mask.sum()
            bw_loss = (self.nll_loss(self.logsoftmax(lm_bw), target_bw.view(-1)) * mask.contiguous().view(
                -1)).sum() / mask.sum()
        else:
            output, _, mask, length = self._get_rnn_output(input_word, input_char, main_task, mask, hx=hx)
            max_len = length.max()
        target = target[:, :max_len]
        # [batch, length, num_label,  num_label]
        if self.use_crf:
            if self.use_lm:
                if main_task:
                    return self.crf_2.loss(output, target, mask=mask).mean() + self.lm_loss * (fw_loss + bw_loss)
                else:
                    return self.crf_1.loss(output, target, mask=mask).mean() + self.lm_loss * (fw_loss + bw_loss)
            else:
                if main_task:
                    return self.crf_2.loss(output, target, mask=mask).mean()
                else:
                    return self.crf_1.loss(output, target, mask=mask).mean()
        else:
            target = target.contiguous()
            if main_task:
                output = self.dense_softmax_2(output)
            else:
                output = self.dense_softmax_1(output)
            # preds = [batch, length]
            _, preds = torch.max(output[:, :, leading_symbolic:], dim=2)
            preds += leading_symbolic
            output_size = output.size()
            # [batch * length, num_labels]
            output_size = (output_size[0] * output_size[1], output_size[2])
            output = output.view(output_size)
            if self.use_lm:
                return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / \
                       mask.sum() + self.lm_loss * (fw_loss + bw_loss), preds
            else:
                return (self.nll_loss(self.logsoftmax(output), target.view(-1)) * mask.contiguous().view(-1)).sum() / \
                       mask.sum(), preds

    def decode(self, input_word, input_char, target, main_task, mask, hx=None, leading_symbolic=0):
        # output from rnn [batch, length, tag_space]
        if self.use_lm:
            output, _, mask, length, lm_fw, lm_bw = self._get_rnn_output(input_word, input_char, main_task,
                                                                                 mask, hx=hx)
        else:
            output, _, mask, length,  = self._get_rnn_output(input_word, input_char, main_task, mask, hx=hx)
        max_len = length.max()
        target = target[:, :max_len]
        if main_task:
            preds = self.crf_2.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        else:
            preds = self.crf_1.decode(output, mask=mask, leading_symbolic=leading_symbolic)
        return preds, (torch.eq(preds, target.data).float() * mask.data).sum()