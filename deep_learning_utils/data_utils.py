from typing import List, Tuple, Dict, Optional, Iterable, Any

import torch
import numpy as np

from typeguard import typechecked # https://pypi.org/project/typeguard/

#%%----------------------------------------------------------------------------
class SequenceDataWithLabels(torch.utils.data.Dataset):
    """
    Parameters
    ----------
    token_IDs: Iterable[Iterable[int]]
        Token IDs of each sentence. For example::

            [
                [101, 1253, 2341, 55234, 102],           # Sentence No.1
                [101, 2425, 1212, 34563, 1102, 102],     # Sentence No.2
                [101, 122, 210, 102],                    # Sentence No.3
                [101, 2231, 402, 10, 87, 32, 230, 102],  # Sentence No.4
            ]

    labels: Iterable[int] or None
        The labels of each sentence. (For example, the positive/negative label
        in a sentence sentiment modeling task.)
    """
    @typechecked
    def __init__(
            self,
            token_IDs: Iterable[Iterable[int]],
            labels: Optional[Iterable[int]] = None
            ):
        _check_length(token_IDs, labels)
        if labels is None:
            labels = [-1] * len(token_IDs)
        # END IF
        self.data_with_labels = list(zip(*[token_IDs, labels]))

    def __getitem__(self, i):
        return self.data_with_labels[i]

    def __len__(self):
        return len(self.data_with_labels)

#%%----------------------------------------------------------------------------
@typechecked
def zip_and_collate(token_IDs_with_labels: List[Tuple[List[int], int]]):
    """
    This function transposes (i.e., "zips") the input data a list of token IDs
    (i.e., "features") and a list of labels, and then calculates the padded
    IDs and masks.

    This function is intended to be used as the ``collate_fn`` of
    ``torch.utils.data.DataLoader``.

    Parameters
    ----------
    token_IDs_with_labels : List[Tuple[List[int], int]]
        The "packed" input data. It should have the following structure::

            [
                ([101, 1253, 2341, 55234, 102],          0),
                ([101, 2425, 1212, 34563, 1102, 102],    1),
                ([101, 122, 210, 102],                   0),
                ([101, 2231, 402, 10, 87, 32, 230, 102], 0),
            ]

    Returns
    -------
    result : Dict[str, torch.Tensor]
        The padded token IDs, the masks, and the labels. It is a dictionary
        that looks like this::

            {"padded_token_IDs": padded_IDs,
             "masks": masks,
             "labels": labels}
    """
    token_IDs, labels = zip(*token_IDs_with_labels)
    return pad_and_mask(token_IDs, labels=labels)

#%%----------------------------------------------------------------------------
@typechecked
def pad_and_mask(token_IDs: Iterable[Iterable[int]], *,
                  labels: Optional[Iterable[int]] = None) -> Dict:
    """
    Pads ``token_IDs`` to have the same length (with 0's), calculates the masks
    (i.e., use 1 to represent non-padded IDs and use 0 to represent padded
    locations).

    Parameters
    ----------
    token_IDs: Iterable[Iterable[int]]
        Token IDs of each sentence. For example::

            [
                [101, 1253, 2341, 55234, 102],           # Sentence No.1
                [101, 2425, 1212, 34563, 1102, 102],     # Sentence No.2
                [101, 122, 210, 102],                    # Sentence No.3
                [101, 2231, 402, 10, 87, 32, 230, 102],  # Sentence No.4
            ]

    labels: Iterable[int] or None
        The labels of each sentence. (For example, the positive/negative label
        in a sentence sentiment modeling task.)

    Returns
    -------
    result : Dict[str, torch.Tensor]
        The padded token IDs, the masks, and the labels. It is a dictionary
        that looks like this::

            {"padded_token_IDs": padded_IDs,
             "masks": masks,
             "labels": labels}

    Reference
    ---------
    https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    """
    _check_length(token_IDs, labels)

    padded_IDs = _pad(token_IDs)
    masks = np.where(padded_IDs != 0, 1, 0)

    padded_IDs_ = torch.tensor(padded_IDs, dtype=torch.int64)
    masks_ = torch.tensor(masks, dtype=torch.int64)

    if labels is None:
        labels_ = torch.tensor([-1] * len(padded_IDs), dtype=torch.int64)
    else:
        labels_ = torch.tensor(labels, dtype=torch.int64)
    # END IF-ELSE

    result = {'padded_token_IDs': padded_IDs_,
              'masks': masks_,
              'labels': labels_}

    return result

#%%----------------------------------------------------------------------------
@typechecked
def _pad(list_of_lists: Iterable[Iterable[Any]], empty_token: Any = 0):
    max_length = max([len(sub_list) for sub_list in list_of_lists])

    padded = []
    for sub_list in list_of_lists:
        sub_list = list(sub_list)  # sometimes it could be a tuple
        remaining_length = max_length - len(sub_list)
        sub_list_padded = sub_list + [0] * (remaining_length)
        padded.append(sub_list_padded)
    # END FOR
    return np.array(padded)

#%%----------------------------------------------------------------------------
def _check_length(token_IDs: Iterable[Any], labels: Optional[Iterable[Any]]):
    """
    Checks that ``token_IDs`` and ``labels`` have consistent length.
    """
    if len(token_IDs) == 0:
        raise ValueError("Length of `token_IDs` must not be 0.")
    # END IF

    if labels is not None and len(token_IDs) != len(labels):
        raise ValueError("`token_IDs` and `labels` must have the same length.")
    # END IF
