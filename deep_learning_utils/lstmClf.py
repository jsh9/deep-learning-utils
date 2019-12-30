# -*- coding: utf-8 -*-
import torch
import torchtext

from .textCNN import load_pretrained_embedding

class LstmClassifier(torch.nn.Module):
    def __init__(
            self, *,
            vocab: torchtext.vocab.Vocab,
            embedding_dim: int,
            num_hidden: int = 100,
            num_layers: int = 1,
            bidirectional: bool = True,
            num_classes: int = 2,
        ):
        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(len(vocab), embedding_dim)

        self.encoder = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_hidden,
            num_layers=num_layers,
            bidirectional=True
        )

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        m = 4 if bidirectional else 2
        self.decoder = torch.nn.Linear(m * num_hidden, num_classes)

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
        self.embedding.weight.requires_grad = False

    def forward(self, inputs: torch.Tensor):
        # inputs的形状是(批量大小, 词数)，因为LSTM需要将序列长度(seq_len)作为第一维，所以将输入转置后
        # 再提取词特征，输出形状为(词数, 批量大小, 词向量维度)
        embeddings = self.embedding(inputs.permute(1, 0))
        # rnn.LSTM只传入输入embeddings，因此只返回最后一层的隐藏层在各时间步的隐藏状态。
        # outputs形状是(词数, 批量大小, 2 * 隐藏单元个数)
        outputs, _ = self.encoder(embeddings) # output, (h, c)
        # 连结初始时间步和最终时间步的隐藏状态作为全连接层输入。它的形状为
        # (批量大小, 4 * 隐藏单元个数)。
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        output = self.decoder(encoding)
        return output
