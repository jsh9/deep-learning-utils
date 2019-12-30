# -*- coding: utf-8 -*-
"""
Adapted from https://github.com/ShusenTang/Dive-into-DL-PyTorch/blob/1a1a4677549e402b60f4a395050fd07f8b6e19f6/code/chapter10_natural-language-processing/10.8_sentiment-analysis-cnn.ipynb
"""

from typing import List, Dict

import torch
import torchtext
import torch.nn.functional as F

import typeguard

from .data_util_classes import FeatureLabelOptionPack

#%%----------------------------------------------------------------------------
class TextCNNClassifier(torch.nn.Module):
    def __init__(
            self, *,
            vocab: torchtext.vocab.Vocab,
            embedding_dim: int,
            kernel_sizes: List[int] = [3, 4, 5],
            num_channels: List[int] = [100, 100, 100],
            dropout_prob: float = 0.5,
            num_classes: int = 2,
    ):
        typeguard.check_argument_types()

        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(len(vocab), embedding_dim)

        # 不参与训练的嵌入层
        self.constant_embedding = torch.nn.Embedding(len(vocab), embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.decoder = torch.nn.Linear(sum(num_channels), num_classes)

        if len(num_channels) != len(kernel_sizes):
            msg = '`num_channels` and `kernel_sizes` must have the same length.'
            raise ValueError(msg)
        # END IF

        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = torch.nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(
                torch.nn.Conv1d(
                    in_channels = 2 * embedding_dim,
                    out_channels = c,
                    kernel_size = k
                )
            )
        # END FOR

    def populate_embedding_layers_with_pretrained_word_vectors(
            self, pretrained_wordvec: torchtext.vocab.Vectors
    ):
        if self.embedding_dim != pretrained_wordvec.dim:
            msg = 'The dimension of `pretrained_wordvec` should equal `embedding_dim`.'
            raise ValueError(msg)
        # END IF
        embedding_matrix = load_pretrained_embedding(self.vocab.itos, pretrained_wordvec)
        assert(embedding_matrix.shape == (len(self.vocab), self.embedding_dim))
        self.embedding.weight.data.copy_(embedding_matrix)
        self.constant_embedding.weight.data.copy_(embedding_matrix)
        self.constant_embedding.weight.requires_grad = False

    def forward(self, inputs: torch.Tensor):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)),
            dim=2
        )  # (batch, seq_len, 2 * embed_size)

        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)

        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        tmp = [self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs]
        encoding = torch.cat(tmp, dim=1)

        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))

        return outputs

#%%----------------------------------------------------------------------------
class GlobalMaxPool1d(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)

#%%----------------------------------------------------------------------------
def load_pretrained_embedding(
        words: List[str],
        pretrained_wordvec: torchtext.vocab.Vectors
):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_wordvec.vectors[0].shape[0]) # 初始化为0
    num_out_of_vocab_words = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_wordvec.stoi[word]
            embed[i, :] = pretrained_wordvec.vectors[idx]
        except KeyError:
            num_out_of_vocab_words += 1
        # END TRY-EXCEPT
    # END FOR
    if num_out_of_vocab_words > 0:
        print("There are %d out-of-vocabulary words." % num_out_of_vocab_words)
    # END IF
    return embed

#%%----------------------------------------------------------------------------
def unpack_data_for_textCNN(data: Dict) -> FeatureLabelOptionPack:
    typeguard.check_argument_types()
    X = data['padded_token_IDs']
    y = data['labels']
    options = dict()
    flop = FeatureLabelOptionPack(X=X, y=y, options=options)
    return flop
