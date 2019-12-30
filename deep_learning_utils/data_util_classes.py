# -*- coding: utf-8 -*-
import re
import torch
import torchtext
import typeguard
import collections

from typing import Sequence, Optional, Tuple, List, Any, Dict, Union

from . import data_utils

#%%----------------------------------------------------------------------------
class FeatureLabelOptionPack:
    """
    A "feature, label, and option" package.

    Parameters
    ----------
    X : torch.Tensor
        The features.
    y : torch.Tensor
        The labels.
    options : Dict[str, Any]
        The options to pass to the ``forward()`` method of a ``torch.nn.Module``
        model.
    """
    def __init__(self, *, X: torch.Tensor, y: torch.Tensor, options: Dict[str, Any]):
        typeguard.check_argument_types()
        self.X = X
        self.y = y
        self.options = options

    def get_X(self, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """
        Retrieve ``X``.

        Parameters
        ----------
        device : str or torch.device
            Which device (CPU or GPU) to put the data to.

        Returns
        -------
        X : torch.Tensor
        """
        typeguard.check_argument_types()
        X = self.X.to(device)
        return X

    def get_y(self, device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """
        Retrieve ``y``.

        Parameters
        ----------
        device : str or torch.device
            Which device (CPU or GPU) to put the data to.

        Returns
        -------
        y : torch.Tensor
        """
        typeguard.check_argument_types()
        y = self.y.to(device)
        return y

    def get_options(self, device: Union[str, torch.device] = 'cpu') -> dict:
        """
        Retrieve ``options``.

        Parameters
        ----------
        device : str or torch.device
            Which device (CPU or GPU) to put all the tensors of the contents
            of ``options`` to. Non tensor data won't be affected.

        Returns
        -------
        options : Dict[str, Any]
        """
        typeguard.check_argument_types()

        options = self.options
        for key, value in options.items():  # move values to specified device
            if isinstance(value, torch.Tensor):
                options[key] = value.to(device)
            # END IF
        # END FOR

        return options

#%%----------------------------------------------------------------------------
class SequenceDataWithLabels(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    token_IDs: Sequence[Sequence[int]]
        Token IDs of each sentence. For example::

            [
                [101, 1253, 2341, 55234, 102],           # Sentence No.1
                [101, 2425, 1212, 34563, 1102, 102],     # Sentence No.2
                [101, 122, 210, 102],                    # Sentence No.3
                [101, 2231, 402, 10, 87, 32, 230, 102],  # Sentence No.4
            ]

    labels: Sequence[int] or None
        The labels of each sentence. (For example, the positive/negative label
        in a sentence sentiment modeling task.)
    """
    def __init__(
            self,
            token_IDs: Sequence[Sequence[int]],
            labels: Optional[Sequence[int]] = None
    ):
        typeguard.check_argument_types()
        data_utils._check_length(token_IDs, labels)
        if labels is None:
            labels = [-1] * len(token_IDs)
        # END IF
        self.data_with_labels = list(zip(*[token_IDs, labels]))

    def __getitem__(self, i):
        return self.data_with_labels[i]

    def __len__(self):
        return len(self.data_with_labels)

#%%----------------------------------------------------------------------------
class WordTokenizer:
    """
    Tokenize a list of sentences ("``texts``") into look-up indices of each
    word in each sentence.

    This class is useful if you want to train an NLP model from scratch, i.e.,
    without relying on any pre-trained models. Because those pre-trained models
    often have their own tokenizers.

    Parameters
    ----------
    max_length : int or ``None``
        The maximum length you want to keep (or pad to) in the tokenization
        result. If ``None``, the maximum word count of all the sentences of
        ``max_length`` will be used. It can be larger or smaller than the actual
        maximum length of all the sentences. If larger, every sentence will be
        padded. If shorter, some sentences may be truncated.
    min_freq : int, optional
        If a word in ``texts`` appears less than ``min_freq`` times, this
        word will not be included in the vocab object. The default is 5.
        (This parameter has no effect if ``existing_vocab`` is provided.)
    ignore_numbers : bool, optional
        If ``True``, treat numbers in the texts as spaces when tokenizing
        and counting word frequencies. The default is False.
    ignore_special_char : bool, optional
        If ``True``, treat special characters (such as ".", ",", "?") in the
        texts as spaces when tokenizing and counting word frequencies.
        The default is True.
    existing_vocab : torchtext.vocab.Vocab or None
        If provided, it will be used to look up indices of words. If ``None``,
        a new vocab will be built from the incoming text corpus.
    """
    def __init__(
            self,
            max_length: Optional[int] = None,
            min_freq: int = 5,
            ignore_numbers: bool = False,
            ignore_special_char: bool = True,
            existing_vocab: Optional[torchtext.vocab.Vocab] = None
    ):
        typeguard.check_argument_types()
        self.max_length = max_length
        self.min_freq = min_freq
        self.ignore_numbers = ignore_numbers
        self.ignore_special_char = ignore_special_char
        self.existing_vocab = existing_vocab

    def tokenize_texts(
            self, texts: List[str], verbose: bool = True
    ) -> Tuple[List[List[int]], torchtext.vocab.Vocab]:
        """
        Tokenize texts (list of sentences) into token IDs. "Token IDs" is a
        list of lists of integers representing the indices of each word in
        ``texts`` in the vocabulary. The vocabulary is built from all the
        words in ``texts``.

        Parameters
        ----------
        texts : List[str]
            The texts to be tokenized, as a list of sentences.
        verbose : bool
            Whether to show a progress message on the console.

        Returns
        -------
        token_IDs : List[List[int]]
            The token IDs corresponding to each sentence.
        vocab : torchtext.vocab.Vocab
            The vocab object.

        Example
        -------
        .. code-block:: python

            texts = ['I go to school every day',
                     'I like reading and studying math',
                     'My school opens every day',
                     'My reading and studying activities are satisfying']
            tokenizer = data_utils.WordTokenizer(max_length=10, min_freq=2)
            token_IDs, vocab = tokenizer.tokenize_texts(texts)

        Then ``token_IDs`` will be:

        .. code-block:: python

            [[5, 0, 0, 8, 4, 3, 1, 1, 1, 1],  # "1" means pading
             [5, 0, 7, 2, 9, 0, 1, 1, 1, 1],  # "0" means unknown
             [6, 8, 0, 4, 3, 1, 1, 1, 1, 1],
             [6, 7, 2, 9, 0, 0, 0, 1, 1, 1]]
        """
        typeguard.check_argument_types()

        if verbose:
            if self.existing_vocab is None:
                msg = '  Tokenizing texts into token IDs (creating new vocab)... '
            else:
                msg = '  Tokenizing texts into token IDs (using existing vocab)... '
            # END IF-ELSE
            print(msg, end='')
        # END IF

        if self.existing_vocab is None:
            tokenized_texts, vocab = self._tokenize_texts_and_build_vocab(
                texts,
                min_freq=self.min_freq,
                ignore_numbers=self.ignore_numbers,
                ignore_special_char=self.ignore_special_char,
            )  # tokenized_texts: List[List[str]], vocab: torchtext.vocab.Vocab
        else:
            vocab = self.existing_vocab
            tokenized_texts = self._tokenize_texts(
                texts,
                ignore_numbers=self.ignore_numbers,
                ignore_special_char=self.ignore_special_char,
            )
        # END IF-ELSE

        token_IDs = self._create_token_IDs(
            tokenized_texts, vocab, max_length=self.max_length
        )

        if verbose:
            print('done.')
        # END IF

        return token_IDs, vocab

    @classmethod
    def _tokenize_texts_and_build_vocab(
            cls,
            texts: List[str],
            min_freq: int = 5,
            ignore_numbers: bool = False,
            ignore_special_char: bool = False,
    ) -> Tuple[List[List[str]], torchtext.vocab.Vocab]:
        """
        Tokenize ``texts`` (multiple sentences), and build a vocab object.

        Parameters
        ----------
        texts : List[str]
            A list of sentences to tokenize.
        min_freq : int, optional
            If a word in ``texts`` appears less than ``min_freq`` times, this
            word will not be included in the vocab object. The default is 5.
        ignore_numbers : bool, optional
            Whether or not to ignore numbers in the texts when tokenizing
            and counting word frequencies. The default is False.
        ignore_special_char : bool, optional
            Whether or not to ignore special characters (such as ".", ",", "?")
            in the texts when tokenizing and counting word frequencies.
            The default is False.

        Returns
        -------
        tokenized : List[List[str]]
            The tokenized text, i.e., each string in ``texts`` are broken into
            a list of words.
        vocab : torchtext.vocab.Vocab
            The vocab object built from the words in ``texts``.
        """
        tokenized = cls._tokenize_texts(
            texts,
            ignore_numbers=ignore_numbers,
            ignore_special_char=ignore_special_char
        )

        flattened = cls._flatten(tokenized)
        word_counts = collections.Counter(flattened)
        vocab = torchtext.vocab.Vocab(  # <unk> & <pad> will be auto injected
            word_counts, min_freq=min_freq
        )

        return tokenized, vocab

    @classmethod
    def _tokenize_texts(
            cls,
            texts: List[str],
            ignore_numbers: bool = False,
            ignore_special_char: bool = False,
    ) -> List[List[str]]:
        """
        Tokenize ``texts`` (multiple sentences).

        Parameters
        ----------
        texts : List[str]
            A list of sentences to tokenize.
        ignore_numbers : bool, optional
            Whether or not to ignore numbers in the texts when tokenizing
            and counting word frequencies. The default is False.
        ignore_special_char : bool, optional
            Whether or not to ignore special characters (such as ".", ",", "?")
            in the texts when tokenizing and counting word frequencies.
            The default is False.

        Returns
        -------
        tokenized : List[List[str]]
            The tokenized text, i.e., each string in ``texts`` are broken into
            a list of words.
        """
        tokenized = []  # it will be a List[List[str]]
        for text in texts:
            if ignore_numbers and ignore_special_char:
                text = re.sub('[^A-za-z]+', ' ', text)
            elif ignore_numbers and not ignore_special_char:
                text = re.sub('[^0-9]+', ' ', text)
            elif not ignore_numbers and ignore_special_char:
                text = re.sub('[^A-za-z0-9]+', ' ', text)
            else:
                pass
            # END IF-ELSE
            words = text.lower().split()
            tokenized.append(words)
        # END FOR
        return tokenized

    @classmethod
    def _create_token_IDs(
            cls,
            tokenized_texts: List[List[str]],
            vocab: torchtext.vocab.Vocab,
            max_length: Optional[int] = None
    ) -> List[List[int]]:
        """
        Create token IDs. For example, from [['hello', 'world'], ['good', 'bye']]
        to [[231, 342], [352, 31]]

        Parameters
        ----------
        tokenized_texts : List[List[str]]
            Texts that are already tokenized (i.e., broken into words).
        vocab : torchtext.vocab.Vocab
            The vocab object that is used to look up the index of each word.
        max_length : int or ``None``
            The maximum length of each token ID list. If ``None``, the maximum
            length of each sub-list of ``tokenized_texts`` will be used.

        Returns
        -------
        token_IDs : List[List[int]]
            Token IDs.
        """
        raw_token_IDs = []  # un-padded and un-truncated, List[List[int]]
        for words_in_this_sentence in tokenized_texts:
            IDs = [vocab.stoi[word] for word in words_in_this_sentence]
            raw_token_IDs.append(IDs)
        # END FOR
        padding_value = vocab.stoi['<pad>']
        token_IDs = cls._pad_or_truncate(
            raw_token_IDs, padding_value, max_length=max_length
        )
        return token_IDs

    @staticmethod
    def _flatten(texts: List[List[str]]) -> List[str]:
        """
        Flatten texts such as [['hello', 'world'], ['good', 'morning']] into
        ['hello', 'world', 'good', 'morning']

        References
        ----------
        https://stackoverflow.com/a/952952
        """
        flattened = [item for sublist in texts for item in sublist]
        return flattened

    @staticmethod
    def _pad_or_truncate(
            list_: List[List[int]],
            padding_value: int,
            max_length: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Pad or truncate each element in ``list_`` to have ``max_length``.

        Parameters
        ----------
        list_ : List[List[int]]
            The list.
        padding_value : int
            What value to use to pad shorter sub-lists.
        max_length : int, optional
            The maximum length desired. If ``None``, it will be the maximum
            length of the sub-lists.

        Returns
        -------
        new_list : List[List[int]]
            The newly constructed list, whose every sub-list has a length of
            ``max_length``.
        """
        if max_length is None:
            max_length = max([len(sublist) for sublist in list_])
        # END IF

        new_list = []
        for sublist in list_:
            if len(sublist) > max_length:
                sublist_ = sublist[:max_length]  # truncate
            else:
                sublist_ = sublist + [padding_value] * (max_length - len(sublist))
            # END FOR
            new_list.append(sublist_)
        # END FOR
        return new_list
