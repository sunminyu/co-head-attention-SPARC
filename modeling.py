# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math

import numpy
import six
import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits, embedding, softmax
# from roberta.modeling_roberta import RobertaModel
import collaborative_attention as co_attention

NO_ANS = -1


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).
        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort
        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.sort(-x_len)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BERTEmbeddings(nn.Module):
    #??????3???embedding?????????????????????????????????+dropout?????????
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states) 
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SparseAttention(nn.Module):
    def __init__(self, config, num_sparse_heads):
        super(SparseAttention, self).__init__()
        if config.hidden_size % num_sparse_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, num_sparse_heads))
        self.num_attention_heads = num_sparse_heads
        self.attention_head_size = int(config.hidden_size / num_sparse_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.shifted_gelu = lambda x: gelu(x) + 0.2 # makes all positive

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, input_ids, ngram=1):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [B, h, T, d/h]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [B, h, T, d/h]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, h, T, T]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(-1)

        # Normalize the attention scores to probabilities.
        # attention_probs = gelu(attention_scores)
        attention_probs = self.relu(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        context_layer = attention_probs.transpose(1, 2)  # [B, T, h, T]
        return context_layer


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, masks):
        all_encoder_layers = []
        if not isinstance(masks, list):
            masks = [masks] * len(self.layer)
        for layer_module, attention_mask in zip(self.layer, masks):
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # In this case, we create [batch_size, 1, from_seq_length, to_seq_length]
            elif attention_mask.dim() == 3:
                extended_attention_mask = attention_mask.unsqueeze(1)
            elif attention_mask.dim() == 4:
                extended_attention_mask = attention_mask
            else:
                raise ValueError(attention_mask.dim())

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.float()
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            hidden_states = layer_module(hidden_states, extended_attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForQuestionAnswering(nn.Module):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits


class BertNoAnswer(nn.Module):
    def __init__(self, hidden_size, context_length=317):
        super(BertNoAnswer, self).__init__()
        self.context_length = context_length
        self.W_no = nn.Linear(hidden_size, 1)
        self.no_answer = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, 2))

    def forward(self, sequence_output, start_logit, end_logit, mask=None):
        if mask is None:
            nbatch, length, _ = sequence_output.size()
            mask = torch.ones(nbatch, length)
        mask = mask.float()
        mask = mask.unsqueeze(-1)[:, 1:self.context_length + 1]
        mask = (1.0 - mask) * -10000.0
        sequence_output = sequence_output[:, 1:self.context_length + 1]
        start_logit = start_logit[:, 1:self.context_length + 1] + mask
        end_logit = end_logit[:, 1:self.context_length + 1] + mask

        # No-answer option
        pa_1 = nn.functional.softmax(start_logit.transpose(1, 2), -1)  # B,1,T
        v1 = torch.bmm(pa_1, sequence_output).squeeze(1)  # B,H
        pa_2 = nn.functional.softmax(end_logit.transpose(1, 2), -1)  # B,1,T
        v2 = torch.bmm(pa_2, sequence_output).squeeze(1)  # B,H
        pa_3 = self.W_no(sequence_output) + mask
        pa_3 = nn.functional.softmax(pa_3.transpose(1, 2), -1)  # B,1,T
        v3 = torch.bmm(pa_3, sequence_output).squeeze(1)  # B,H

        bias = self.no_answer(torch.cat([v1, v2, v3], -1))  # B,1

        return bias


class BertForSQuAD2(nn.Module):
    def __init__(self, config, context_length=317):
        super(BertForSQuAD2, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.na_head = BertNoAnswer(config.hidden_size, context_length)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, labels=None):
        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        span_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        na_logits = self.na_head(sequence_output, start_logits, end_logits, attention_mask)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = (na_logits + logits) / 2  # mean

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            unanswerable_loss = loss_fct(logits, labels)
            span_loss = (start_loss + end_loss) / 2
            total_loss = span_loss + unanswerable_loss
            return total_loss
        else:
            probs = nn.functional.softmax(logits, -1)
            _, probs = probs.split(1, dim=-1)
            return start_logits, end_logits, probs


class DenSPI(nn.Module):
    def __init__(self, config,
                 span_vec_size=64,
                 context_layer_idx=-1,
                 question_layer_idx=-1,
                 sparse_ngrams=['1', '2'],
                 use_sparse=True,
                 neg_with_tfidf=False,
                 do_train_filter=False,
                 min_word_id=999):
        super(DenSPI, self).__init__()

        self.dropout=nn.Dropout(0.5)#????????????0.5???
        # Dense modules
        self.bert = BertModel(config)
        self.bert_q = self.bert

        # lstm layer
        # self.lstm=DynamicLSTM(config.hidden_size, 384, num_layers=1, batch_first=True, bidirectional=True, rnn_type='LSTM')

        # Sparse modules (Sparc)
        self.sparse_start = nn.ModuleDict({
            key: co_attention.Co_SparseAttention(config, num_sparse_heads=1)
            for key in sparse_ngrams
        })
        self.sparse_end = nn.ModuleDict({
            key: co_attention.Co_SparseAttention(config, num_sparse_heads=1)
            for key in sparse_ngrams
        })
        self.sparse_start_q = self.sparse_start
        self.sparse_end_q =  self.sparse_end

        # Other parameters
        self.linear = nn.Linear(config.hidden_size, config.hidden_size) # For filter
        self.default_value = nn.Parameter(torch.randn(1))
        self.tfidf_weight = nn.Parameter(torch.randn(1))
        self.min_word_id = min_word_id

        # Arguments
        self.span_vec_size = span_vec_size
        self.context_layer_idx = context_layer_idx
        self.question_layer_idx = question_layer_idx
        self.do_train_filter = do_train_filter
        self.use_sparse = use_sparse
        self.sparse_ngrams = sparse_ngrams
        self.sigmoid = nn.Sigmoid()
        self.neg_with_tfidf = neg_with_tfidf

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            # if isinstance(module, nn.Linear):
            #     module.bias.data.zero_()

        self.apply(init_weights)

    def forward(self, input_ids=None, input_mask=None, query_ids=None, query_mask=None,
                start_positions=None, end_positions=None, neg_context_ids=None, neg_context_mask=None,
                pos_score=None, neg_score=None):

        #??????????????????============LSTM===================
        # feature_len = torch.sum(input_mask, dim=-1)============LSTM
        # print(feature_len)============LSTM
        # a = np.array([256])============LSTM
        # a = torch.from_numpy(a)============LSTM
        # a = a.cuda()============LSTM
        # print(a)============LSTM
        if input_ids is not None:
            bs, seq_len = input_ids.size()

            # BERT reps
            context_layers, _ = self.bert(input_ids, None, input_mask)
            context_layer_all = context_layers[self.context_layer_idx]
            # print("``````````````````````````see here````````````````````````")============LSTM
            # print(np.array(context_layers).shape)============LSTM
            # v = self.dropout(context_layers)============LSTM
            # pre, (_, _) = self.lstm(context_layers[self.context_layer_idx], a)============LSTM
            # print("lstm")============LSTM
            # print(pre.shape)============LSTM
            # #add============LSTM
            # context_layer_all = pre============LSTM
            # print("bert_out")============LSTM
            # print(context_layer_all.shape)============LSTM


            # Append negative doc/ids
            if neg_context_ids is not None:
                assert neg_context_mask is not None and len(neg_context_ids.size()) == 2
                context_layers2, _ = self.bert(neg_context_ids, None, neg_context_mask)
                context_layer2_all = context_layers2[self.context_layer_idx]
                context_layer_all = torch.cat([context_layer_all, context_layer2_all], 1)
                input_mask = torch.cat([input_mask, neg_context_mask], 1)
                input_ids = torch.cat([input_ids, neg_context_ids], 1)
                bs, seq_len = input_ids.size()

            # Calculate dense logits
            context_layer = context_layer_all[:, :, :-self.span_vec_size]
            span_layer = context_layer_all[:, :, -self.span_vec_size:]
            start, end, = context_layer.chunk(2, dim=2)
            span_start, span_end = span_layer.chunk(2, dim=2)
            span_logits = span_start.matmul(span_end.transpose(1, 2))

            # Calculate Sparc
            start_sps = {}
            end_sps = {}
            sparse_mask = (input_ids >= self.min_word_id).float()
            input_diag = (1 - torch.diag(torch.ones(input_ids.shape[1]))).to(sparse_mask.get_device())
            for ngram in self.sparse_ngrams:
                start_sps[ngram] = self.sparse_start[ngram](
                    context_layer_all,
                    (1 - input_mask).float() * -1e9,
                    input_ids, ngram=self.sparse_ngrams,
                )
                end_sps[ngram] = self.sparse_end[ngram](
                    context_layer_all,
                    (1 - input_mask).float() * -1e9,
                    input_ids, ngram=self.sparse_ngrams,
                )
                start_sps[ngram] = start_sps[ngram][:,:,0,:] * sparse_mask.unsqueeze(1) * input_diag.unsqueeze(0)
                end_sps[ngram] = end_sps[ngram][:,:,0,:] * sparse_mask.unsqueeze(1) * input_diag.unsqueeze(0)

            # Filter calculation
            filter_start_logits, filter_end_logits = self.linear(context_layer_all).chunk(2, dim=2)
            filter_start_logits = filter_start_logits[:,:,0]
            filter_end_logits = filter_end_logits[:,:,0]

            # Embed context
            if query_ids is None:
                return start, end, span_logits, filter_start_logits, filter_end_logits, start_sps, end_sps

        if query_ids is not None:
            # BERT reps
            question_layers, _ = self.bert_q(query_ids, None, query_mask)
            question_layer = question_layers[self.question_layer_idx][:, :, :-self.span_vec_size]
            query_start, query_end = question_layer[:, :1, :].chunk(2, dim=2)  # Just [CLS]

            # For query-side Sparc
            q_start_sps = {}
            q_end_sps = {}
            query_sparse_mask = ((query_ids >= self.min_word_id) & (query_ids != 1029)).float()
            for ngram in self.sparse_ngrams:
                q_start_sps[ngram] = self.sparse_start_q[ngram](
                    question_layers[self.question_layer_idx],
                    (1 - query_mask).float() * -1e9,
                    query_ids, ngram=self.sparse_ngrams
                )
                q_end_sps[ngram] = self.sparse_end_q[ngram](
                    question_layers[self.question_layer_idx],
                    (1 - query_mask).float() * -1e9,
                    query_ids, ngram=self.sparse_ngrams
                )
                q_start_sps[ngram] = q_start_sps[ngram][:,0,0,:] * query_sparse_mask
                q_end_sps[ngram] = q_end_sps[ngram][:,0,0,:] * query_sparse_mask

            # Embed question
            if input_ids is None:
                return query_start, query_end, q_start_sps, q_end_sps

        # After this line, we calculate logits and loss
        start_logits = start.matmul(query_start.transpose(1, 2)).squeeze(-1)
        end_logits = end.matmul(query_end.transpose(1, 2)).squeeze(-1)
        dense_logits = start_logits.unsqueeze(2) + end_logits.unsqueeze(1) + span_logits

        # Get sparse logits (kernelized)
        sp_logits = None
        if self.use_sparse:
            sp_logits_list = []
            input_ids_list = [input_ids]
            sparse_mask_list = [sparse_mask]
            start_logits = start_logits.unsqueeze(1)

            for neg_idx, (_input_ids, _sparse_mask, _start_logits) in enumerate(
                    zip(input_ids_list, sparse_mask_list, start_logits.unbind(1))):
                sp_start_logits = torch.zeros_like(_start_logits).to(_start_logits.device)
                sp_end_logits = torch.zeros_like(_start_logits).to(_start_logits.device)

                # logits for unigram Sparc
                if '1' in self.sparse_ngrams:
                    mxq = (_input_ids.unsqueeze(2) == query_ids.unsqueeze(1)) * (
                        _sparse_mask.unsqueeze(2).byte() * query_sparse_mask.unsqueeze(1).byte())
                    _start_sps = start_sps['1'].split(seq_len, dim=1)[neg_idx]
                    _end_sps = end_sps['1'].split(seq_len, dim=1)[neg_idx]

                    sp_start_logits += (_start_sps.matmul(mxq.float()).matmul(
                        q_start_sps['1'].unsqueeze(2))).squeeze(2) * _sparse_mask
                    sp_end_logits += (_end_sps.matmul(mxq.float()).matmul(
                        q_end_sps['1'].unsqueeze(2))).squeeze(2) * _sparse_mask

                # logits for bigram Sparc
                if '2' in self.sparse_ngrams:
                    bi_ids = torch.cat(
                        [_input_ids[:,:-1].unsqueeze(2), _input_ids[:,1:].unsqueeze(2)], 2
                    )
                    bi_qids = torch.cat(
                        [query_ids[:,:-1].unsqueeze(2), query_ids[:,1:].unsqueeze(2)], 2
                    )
                    bi_sparse_mask = (_sparse_mask[:,:-1].byte() & _sparse_mask[:,1:].byte())
                    bi_query_sparse_mask = (query_sparse_mask[:,:-1].byte() & query_sparse_mask[:,1:].byte())
                    bi_mxq = (bi_ids.unsqueeze(2) == bi_qids.unsqueeze(1)) * (
                            bi_sparse_mask.unsqueeze(2).unsqueeze(3) *
                            bi_query_sparse_mask.unsqueeze(1).unsqueeze(3))
                    bi_mxq = bi_mxq.sum(-1) == 2
                    _start_sps = start_sps['2'].split(seq_len, dim=1)[neg_idx]
                    _end_sps = end_sps['2'].split(seq_len, dim=1)[neg_idx]

                    sp_start_logits[:,:-1] += (_start_sps[:,:-1,:-1].matmul(bi_mxq.float()).matmul(
                        q_start_sps['2'][:,:-1].unsqueeze(2))).squeeze(2) * bi_sparse_mask.float()
                    sp_end_logits[:,:-1] += (_end_sps[:,:-1,:-1].matmul(bi_mxq.float()).matmul(
                        q_end_sps['2'][:,:-1].unsqueeze(2))).squeeze(2) * bi_sparse_mask.float()

                # logits for trigram Sparc (Not used)
                if '3' in self.sparse_ngrams:
                    tri_ids = torch.cat(
                            [_input_ids[:,:-2].unsqueeze(2), _input_ids[:,1:-1].unsqueeze(2),
                                _input_ids[:,2:].unsqueeze(2)], 2
                    )
                    tri_qids = torch.cat(
                        [query_ids[:,:-2].unsqueeze(2), query_ids[:,1:-1].unsqueeze(2),
                            query_ids[:,2:].unsqueeze(2)], 2
                    )
                    tri_sparse_mask = (_sparse_mask[:,:-2].byte() & _sparse_mask[:,1:-1].byte() & _sparse_mask[:,2:].byte())
                    tri_query_sparse_mask = (query_sparse_mask[:,:-2].byte() & query_sparse_mask[:,1:-1].byte() &
                        query_sparse_mask[:,2:].byte())
                    tri_mxq = (tri_ids.unsqueeze(2) == tri_qids.unsqueeze(1)) * (
                            tri_sparse_mask.unsqueeze(2).unsqueeze(3) *
                            tri_query_sparse_mask.unsqueeze(1).unsqueeze(3))
                    tri_mxq = tri_mxq.sum(-1) == 3
                    _start_sps = start_sps['3'].split(seq_len, dim=1)[neg_idx]
                    _end_sps = end_sps['3'].split(seq_len, dim=1)[neg_idx]

                    sp_start_logits[:,:-2] += (_start_sps[:,:-2,:-2].matmul(tri_mxq.float()).matmul(
                        q_start_sps['3'][:,:-2].unsqueeze(2))).squeeze(2) * tri_sparse_mask.float()
                    sp_end_logits[:,:-2] += (_end_sps[:,:-2,:-2].matmul(tri_mxq.float()).matmul(
                        q_end_sps['3'][:,:-2].unsqueeze(2))).squeeze(2) * tri_sparse_mask.float()

                # Sparse logits
                sp_logits = sp_start_logits.unsqueeze(2) + sp_end_logits.unsqueeze(1)
                sp_logits_list.append(sp_logits)
            sp_logits = torch.cat(sp_logits_list, dim=1)

        # Aggregate or not
        if sp_logits is not None:
            all_logits = dense_logits + sp_logits
        else:
            all_logits = dense_logits

        # Calculate loss
        if start_positions is not None and end_positions is not None:
            if self.neg_with_tfidf:
                full_dim = all_logits.size(-1)
                all_logits[:,full_dim//2:] += neg_score.unsqueeze(1).unsqueeze(1) * self.sigmoid(self.tfidf_weight)

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1 and start_positions.size(-1) == 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1 and end_positions.size(-1) == 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = seq_len
            span_ignored_index = ignored_index ** 2
            start_positions.clamp_(NO_ANS, seq_len)
            end_positions.clamp_(NO_ANS, seq_len)
            cel_1d = CrossEntropyLossWithDefault(default_value=self.default_value,
                                                 ignore_index=seq_len, reduction='none')
            cel_2d = CrossEntropyLossWithDefault(default_value=self.default_value,
                                                 ignore_index=span_ignored_index, reduction='none')
            span_target = start_positions * ignored_index + end_positions

            # needed to handle -1
            span_target.clamp_(NO_ANS, span_ignored_index)
            valid = (start_positions < seq_len) & (end_positions < seq_len)
            span_target = valid.long() * span_target + (1 - valid.long()) * span_ignored_index

            if len(start_positions.size()) == 1:
                # Start/end prediction loss
                help_loss = 0.5 * (cel_1d(all_logits.mean(2), start_positions).mean() +
                                   cel_1d(all_logits.mean(1), end_positions).mean())
                d_help_loss = 0.5 * (cel_1d(dense_logits.mean(2), start_positions).mean() +
                                   cel_1d(dense_logits.mean(1), end_positions).mean())

                # Span prediction loss
                true_loss = cel_2d(all_logits.view(all_logits.size(0), -1), span_target).mean()
                d_true_loss = cel_2d(dense_logits.view(dense_logits.size(0), -1), span_target).mean()
            else:
                raise NotImplementedError()

            # Span + start/end loss
            loss = true_loss + help_loss

            # Dense only loss
            if True:
                d_loss = d_true_loss + d_help_loss
                loss = loss + d_loss

            # Filter loss
            filter_loss = None
            if self.do_train_filter:
                length = torch.tensor(all_logits.size(-1)).to(start_logits.device)
                eye = torch.eye(length + 2).to(start_logits.device)
                start_1hot = embedding(start_positions + 1, eye)[:, 1:-1]
                end_1hot = embedding(end_positions + 1, eye)[:, 1:-1]
                start_loss = binary_cross_entropy_with_logits(filter_start_logits, start_1hot, pos_weight=length)
                end_loss = binary_cross_entropy_with_logits(filter_end_logits, end_1hot, pos_weight=length)
                filter_loss = 0.5 * start_loss + 0.5 * end_loss

            return loss, filter_loss, all_logits, filter_start_logits, filter_end_logits
        else:
            return all_logits, filter_start_logits, filter_end_logits


class CrossEntropyLossWithDefault(nn.CrossEntropyLoss):
    def __init__(self, default_value=None, ignore_index=-100, **kwargs):
        if ignore_index >= 0:
            ignore_index += 1
        super(CrossEntropyLossWithDefault, self).__init__(ignore_index=ignore_index, **kwargs)
        if default_value is None:
            self.default_value = nn.Parameter(torch.randn(1))
        else:
            self.default_value = default_value

    def forward(self, input_, target):
        assert len(input_.size()) == 2
        default_value = self.default_value.unsqueeze(0).repeat(input_.size(0), 1)
        new_input = torch.cat([default_value, input_], 1)
        new_target = target + 1
        assert new_target.min().item() >= 0, (new_target.min().item(), target.min().item())
        loss = super(CrossEntropyLossWithDefault, self).forward(new_input, new_target)
        return loss
