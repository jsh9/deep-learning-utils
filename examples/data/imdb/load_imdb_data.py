# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 16:42:46 2020
"""
import os
import sys
import pickle
import tarfile
import urllib.request
from tqdm import tqdm

from typing import Dict, List, Tuple

import typeguard

def load_imdb_sentiment_data(
        cache_dir: str,
        save_pickle_if_processed: bool = True,
        verbose: bool = True
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Load IMDb sentiment data (https://ai.stanford.edu/~amaas/data/sentiment/).

    Parameters
    ----------
    cache_dir : str
        The directory for you to cache your gz file, folder, and pickle file.
    save_pickle_if_processed : bool, optional
        Whether to save a pickle file containing processed data. Default = True.
    verbose : bool, optional
        Whether to show progress on the console. The default is True.

    Returns
    -------
    data : Dict[str, List[Tuple[str, int]]]
        The IMDb sentiment data. It has the following structure::

            {"train": [
                         ("This movie is great!", 1),
                         ("This movie is terrible!", 0),
                         ...
                      ]
             "test": [
                         ("This movie is fantastic!", 1),
                         ("Meh", 0),
                         ...
                     ]
            }
    """
    typeguard.check_argument_types()

    #----------- Extract tar.gz file or load from extracted folder ------------
    aclImdb_folder_full = os.path.join(cache_dir, "aclImdb")
    if not os.path.exists(aclImdb_folder_full):

        #------------------ Download or load tar.gz file ----------------------
        gz_name = "aclImdb_v1.tar.gz"
        if not os.path.exists(os.path.join(cache_dir, gz_name)):
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            # END IF
            print_(f"Downloading {gz_name} from {url}... ", verbose=verbose, end="")
            urllib.request.urlretrieve(url, os.path.join(cache_dir, gz_name))
            print_("done.", verbose=verbose)
        else:
            if verbose:
                print(f"{gz_name} loaded from hard drive.")
            # END IF
        # END IF-ELSE
        #----------------------------------------------------------------------

        print_(f"Extracting from {gz_name}... ", verbose=verbose, end="")
        tar = tarfile.open(os.path.join(cache_dir, gz_name))
        tar.extractall(cache_dir)
        tar.close()
        print_("done.", verbose=verbose)
    else:
        print_(f"Folder {aclImdb_folder_full} found.", verbose=verbose)
    # END IF-ELSE

    #------------ Load processed pickle, or process ---------------------------
    pickle_filename = "imdb_sentiment.pkl"
    pickle_full_filename = os.path.join(cache_dir, pickle_filename)
    if not os.path.exists(pickle_full_filename):
        print_("Pickle file not found. Process raw data... ", verbose=verbose)
        train = _load_imdb_helper("train", aclImdb_folder_full)
        test = _load_imdb_helper("test", aclImdb_folder_full)
        data = dict()
        data['train'] = train
        data['test'] = test
        print_("Done.")

        if save_pickle_if_processed:
            print_("Saving processed data to pickle...", verbose=verbose, end="")
            with open(pickle_full_filename, "wb") as fp:
                pickle.dump(data, fp)
            # END WITH
            print_("done.", verbose=verbose)
        # END IF
    else:
        print_(f"Pickle file {pickle_filename} found. Loading...", verbose, "")
        with open(pickle_full_filename, "rb") as fp:
            data = pickle.load(fp)
        # END WITH
        print_("done.", verbose=verbose)
    # END ELSE

    typeguard.check_return_type(data)
    return data

def print_(msg: str, verbose: bool = True, end: str = "\n"):
    if verbose:
        print(msg, end=end)
    # END IF

def _load_imdb_helper(folder: str, data_dir: str):
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_dir, folder, label)
        desc = 'Processing raw data: %s/%s' % (folder, label)
        for file in tqdm(os.listdir(folder_name), desc=desc, file=sys.stdout):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append((review, 1 if label == 'pos' else 0))
            # END WITH
        # END FOR
    # END FOR
    return data

if __name__ == "__main__":
    data = load_imdb_sentiment_data("./")
