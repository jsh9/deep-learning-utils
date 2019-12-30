import unittest

import torch
import torchtext
import numpy as np
from deep_learning_utils import data_utils

class Test_Data_Utils(unittest.TestCase):
    def test_WordTokenizer__letters_only(self):
        texts = ['I go to school every day',
                 'I like reading and studying math',
                 'My school opens every day',
                 'My reading and studying activities are satisfying']
        tokenizer = data_utils.WordTokenizer(max_length=10, min_freq=2)
        token_IDs, vocab = tokenizer.tokenize_texts(texts)

        self.assertTrue(isinstance(vocab, torchtext.vocab.Vocab))

        benchmark_token_IDs = [[5, 0, 0, 8, 4, 3, 1, 1, 1, 1],  # "1" means pad
                               [5, 0, 7, 2, 9, 0, 1, 1, 1, 1],
                               [6, 8, 0, 4, 3, 1, 1, 1, 1, 1],
                               [6, 7, 2, 9, 0, 0, 0, 1, 1, 1]]
        self.assertEqual(benchmark_token_IDs, token_IDs)

    def test_WordTokenizer__numbers_and_special_char(self):
        texts = ['I go to school every day',
                 'I like reading and !studying math',  # add "!" before "studying"
                 'My school opens every day',
                 'My reading and studying2 activities are satisfying'] # "2" before "studying"
        tokenizer = data_utils.WordTokenizer(max_length=10, min_freq=2)
        token_IDs, vocab = tokenizer.tokenize_texts(texts)

        self.assertTrue(isinstance(vocab, torchtext.vocab.Vocab))

        benchmark_token_IDs = [[5, 0, 0, 8, 4, 3, 1, 1, 1, 1],  # "1" means pad
                               [5, 0, 7, 2, 0, 0, 1, 1, 1, 1],
                               [6, 8, 0, 4, 3, 1, 1, 1, 1, 1],
                               [6, 7, 2, 0, 0, 0, 0, 1, 1, 1]]
        self.assertEqual(benchmark_token_IDs, token_IDs)

    def test_WordTokenizer__ignore_numbers_and_special_char(self):
        texts = ['I go to school every day',
                 'I like reading and !studying math',  # add "!" before "studying"
                 'My school opens every day',
                 'My reading and studying2 activities are satisfying'] # "2" before "studying"
        tokenizer = data_utils.WordTokenizer(
            max_length=10, min_freq=2,
            ignore_numbers=True, ignore_special_char=True,
        )
        token_IDs, vocab = tokenizer.tokenize_texts(texts)

        self.assertTrue(isinstance(vocab, torchtext.vocab.Vocab))

        benchmark_token_IDs = [[5, 0, 0, 8, 4, 3, 1, 1, 1, 1],  # "1" means pad
                               [5, 0, 7, 2, 9, 0, 1, 1, 1, 1],
                               [6, 8, 0, 4, 3, 1, 1, 1, 1, 1],
                               [6, 7, 2, 9, 0, 0, 0, 1, 1, 1]]
        self.assertEqual(benchmark_token_IDs, token_IDs)

    def test_pad_and_mask__with_labels(self):
        X = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5]]
        y = [0, 1, 2]

        result = data_utils.pad_and_mask(X, labels=y)
        self.assertTrue(
            np.allclose([[1, 2, 3, 4, 5],
                         [2, 3, 4, 5, 0],
                         [3, 4, 5, 0, 0]],
            result['padded_token_IDs'])
        )
        self.assertTrue(
            np.allclose([[1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0]],
            result['masks'])
        )
        self.assertTrue(np.allclose([0, 1, 2], result['labels']))
        self.assertEqual(3, len(result))

    def test_pad_and_mask__without_labels(self):
        X = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5]]

        result = data_utils.pad_and_mask(X, labels=None)
        self.assertTrue(
            np.allclose([[1, 2, 3, 4, 5],
                         [2, 3, 4, 5, 0],
                         [3, 4, 5, 0, 0]],
            result['padded_token_IDs'])
        )
        self.assertTrue(
            np.allclose([[1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0]],
            result['masks'])
        )
        self.assertTrue(np.allclose([-1, -1, -1], result['labels']))
        self.assertEqual(3, len(result))

    def test_create_SequenceDataWithLabels__label_is_not_None(self):
        X = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5]]
        y = [0, 1, 2]
        dataset = data_utils.SequenceDataWithLabels(X, y)
        self.assertEqual(3, len(dataset))
        self.assertEqual(([1, 2, 3, 4, 5], 0), dataset[0])
        self.assertEqual(([2, 3, 4, 5], 1), dataset[1])
        self.assertEqual(([3, 4, 5], 2), dataset[2])

    def test_create_SequenceDataWithLabels__label_is_None(self):
        X = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5]]
        y = None
        dataset = data_utils.SequenceDataWithLabels(X, y)
        self.assertEqual(3, len(dataset))
        self.assertEqual(([1, 2, 3, 4, 5], -1), dataset[0])
        self.assertEqual(([2, 3, 4, 5], -1), dataset[1])
        self.assertEqual(([3, 4, 5], -1), dataset[2])

    def test_use_dataloader_with_SequenceData__label_is_not_None(self):
        X = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5]]
        y = [0, 1, 2]
        dataset = data_utils.SequenceDataWithLabels(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False,
            collate_fn=data_utils.zip_and_collate
        )
        dataloader_ = list(dataloader)
        self.assertEqual(2, len(dataloader_))
        self.assertTrue(
            np.allclose(
                [[1, 2, 3, 4, 5], [2, 3, 4, 5, 0]],
                dataloader_[0]['padded_token_IDs']
            )
        )
        self.assertTrue(
            np.allclose(
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
                dataloader_[0]['masks']
            )
        )
        self.assertTrue(
            np.allclose(
                [0, 1],
                dataloader_[0]['labels']
            )
        )
        self.assertTrue(
            np.allclose(
                [[3, 4, 5]],
                dataloader_[1]['padded_token_IDs']
            )
        )
        self.assertTrue(
            np.allclose(
                [[1, 1, 1]],
                dataloader_[1]['masks']
            )
        )
        self.assertTrue(
            np.allclose(
                [2],
                dataloader_[1]['labels']
            )
        )

    def test_use_dataloader_with_SequenceData__label_is_None(self):
        X = [[1, 2, 3, 4, 5], [2, 3, 4, 5], [3, 4, 5]]
        y = None
        dataset = data_utils.SequenceDataWithLabels(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=False,
            collate_fn=data_utils.zip_and_collate
        )
        dataloader_ = list(dataloader)
        self.assertEqual(2, len(dataloader_))
        self.assertTrue(
            np.allclose(
                [[1, 2, 3, 4, 5], [2, 3, 4, 5, 0]],
                dataloader_[0]['padded_token_IDs']
            )
        )
        self.assertTrue(
            np.allclose(
                [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]],
                dataloader_[0]['masks']
            )
        )
        self.assertTrue(
            np.allclose(
                [-1, -1],
                dataloader_[0]['labels']
            )
        )
        self.assertTrue(
            np.allclose(
                [[3, 4, 5]],
                dataloader_[1]['padded_token_IDs']
            )
        )
        self.assertTrue(
            np.allclose(
                [[1, 1, 1]],
                dataloader_[1]['masks']
            )
        )
        self.assertTrue(
            np.allclose(
                [-1],
                dataloader_[1]['labels']
            )
        )

if __name__ == '__main__':
    SUITE = unittest.TestLoader().loadTestsFromTestCase(Test_Data_Utils)
    unittest.TextTestRunner(verbosity=2).run(SUITE)
