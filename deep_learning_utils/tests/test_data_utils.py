import unittest

import torch
import numpy as np
from deep_learning_utils import data_utils

class Test_Data_Utils(unittest.TestCase):
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
