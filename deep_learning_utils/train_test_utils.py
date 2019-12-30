# -*- coding: utf-8 -*-
import time
import torch
import typeguard
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Callable, Union, Sequence

from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

from .data_util_classes import FeatureLabelOptionPack

TensorTensorTensor = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

#%%----------------------------------------------------------------------------
def unpack_repack_data(data: Sequence[torch.Tensor]) -> FeatureLabelOptionPack:
    """
    Unpack ``data`` and then repack into a ``FeatureLabelOptionPack`` object.

    Parameters
    ----------
    data : Sequence[torch.Tensor]
        The data coming out of a ``DataLoader`` object.

    Returns
    -------
    flop : deep_learning_utils.data_util_classes.FeatureLabelOptionPack
        The repackaged data.
    """
    typeguard.check_argument_types()
    X = data[0]
    y = data[1]
    options = dict()  # no additional options
    flop = FeatureLabelOptionPack(X=X, y=y, options=options)
    return flop

#%%----------------------------------------------------------------------------
def unpack_training_data_for_transformer(data: Dict) -> FeatureLabelOptionPack:
    """
    Unpack ``data`` and then repack into a ``FeatureLabelOptionPack`` object
    specifically for transformer models.

    Parameters
    ----------
    data : Dict
        The data to be unpacked and repackaged. It must be a dictionary that
        has at least three keys: (1) "padded_token_IDs" that contains a PyTorch
        tensor of the token IDs of each word of each sentence, (2) "labels"
        that contains the labels of each sentence, and (3) "masks" that is
        ia PyTorch tensor of 0 and 1 to indicate which positions in
        "padded_token_IDs" are padded (useful for the attention mechanism
        in the transformer models).

    Returns
    -------
    flop : deep_learning_utils.data_util_classes.FeatureLabelOptionPack
        The repackaged data.
    """
    typeguard.check_argument_types()
    X = data['padded_token_IDs']
    y = data['labels']
    masks = data['masks']
    flop = FeatureLabelOptionPack(X=X, y=y, options={'attention_mask': masks})
    return flop

#%%----------------------------------------------------------------------------
def train(*,
        train_iter: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        device: Union[str, torch.device] = "cpu",
        unpack_repack_fn: Callable[[Any], FeatureLabelOptionPack] = unpack_repack_data,
        static_options_to_model: Dict[str, Any] = dict(),
        test_data: Any = None,
        verbose: bool = True,
        verbose_each_batch: bool = True,
        verbose_each_epoch: bool = True,
        eval_each_batch: bool = True,
        eval_test_accuracy: bool = False,
        eval_test_AUC: bool = False,
        eval_test_R2: bool = False,
        plot_learning_curve: bool = True,
) -> None:
    """
    Train a PyTorch model.

    Parameters
    ----------
    train_iter : torch.utils.data.DataLoader
        Training data iteration object. Each item in the iterator must be a
        tuple of two PyTorch tensors ("`X`" and "`y`" respectively).
    model : torch.nn.Module
        A PyTorch neural network model.
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    optimizer : torch.optim.Optimizer
        The optimizer to update the model weights.
    num_epochs : int
        Number of epochs to train the model.
    device : str
        The device to do the training on. It can be "cpu", or "cuda", or other
        valid names (if you have multiple GPUs). Default value is "cpu".
    unpack_repack_fn : Callable[[Any], Dict[str, Any]]
        A function that unpacks each item in ``train_iter`` (or unpacks
        ``test_data``) and repackages the information into a
        :py:class:~`deep_learning_utils.data_util_classes.FeatureLabelOptionPack`
        object. An example function is :py:meth:~`unpack_repack_data`.
    static_options_to_model : Dict[str, Any]
        "Static" keyword arguments to pass to the ``model``'s ``forward()``
        method. "Static" means that these keyword arguments don't change in
        different mini batches or different epochs. You need to pack them into
        a dictionary, whose keys are the argument names and values are argument
        values. It can be an empty dict, which means that no additional
        arguments will be passed to ``forward()``.
    test_data : Any or ``None``
        Test data to evaluate the model performance on. Its type should be
        consistent with what comes out of each iteration of ``train_iter``.
    verbose : bool
        Whether to show messages of progress on the console. Default = True.
    verbose_each_batch : bool
        Whether to print loss values at each batch. Default = True.
    verbose_each_epoch : bool
        Whether to print loss values at each epoch. Default = True.
    eval_each_batch : bool
        If ``True``, evaluate the model on the ``test_data`` at the end of
        every mini batch. Otherwise, only evaluate the model on the
        ``test_data`` at the end of each epoch. This parameter has no effect
        if ``test_data`` is ``None``.
    eval_test_accuracy : bool
        Whether to evaluate test accuracy at the end of each epoch or each
        mini bach. Only effective if the corresponding verbosity is set to
        ``True``. If your modeling problem is not a classification problem,
        there may be runtime errors if you set this to true, or the accuracy
        value you see may not make sense.
    eval_test_AUC: bool
        Whether to evaluate test AUC at the end of each epoch or each
        mini bach. Only effective if the corresponding verbosity is set to
        ``True``. If your modeling problem is not a binary classification
        problem, there may be runtime errors if you set this to true, or the
        AUC value you see may not make sense.
    eval_test_R2: bool
        Whether to evaluate test R2 score at the end of each epoch or each
        mini bach. Only effective if the corresponding verbosity is set to
        ``True``. If your modeling problem is not a regression problem, there
        may be runtime errors if you set this to true, or the R2 value you
        see may not make sense.
    plot_learning_curve : bool
        If ``True``, show a plot of training and testing loss as a function
        of the training process. Note that no testing loss will be shown if
        ``test_data`` is ``None``. And testing loss will only be shown at the
        end of each epoch if ``eval_each_batch`` is set to ``False``.

    Example
    -------
    1. Binary classification

    .. code-block:: python

        import torch
        import numpy as np
        import deep_learning_utils as dlu

        class Classifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(9, 4)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(p=0.2)
                self.layer2 = torch.nn.Linear(4, 2)

            def forward(self, input_):
                x = self.layer1(input_)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.layer2(x)
                return x

        clf = Classifier()

        optimizer = torch.optim.Adam(
            [{"params": clf.layer1.parameters(), "lr": 0},
             {"params": clf.layer2.parameters(), "lr": 1e-3}]
        )

        X = torch.rand(124, 9)
        y = torch.tensor(np.random.rand(124) > 0.5, dtype=int)
        train_data = torch.utils.data.TensorDataset(X, y)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)

        X_test = torch.rand(73, 9)
        y_test = torch.tensor(np.random.rand(73) > 0.5, dtype=int)

        loss_fn = torch.nn.CrossEntropyLoss()

        dlu.train_test_utils.train(train_iter, model=clf, loss_fn=loss_fn,
                                   optimizer=optimizer, num_epochs=20,
                                   test_data=(X_test, y_test),
                                   eval_test_accuracy=True, eval_test_AUC=True)


    2. Regression

    .. code-block:: python

        import torch
        import deep_learning_utils as dlu

        class Regressor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(9, 4)
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(p=0.2)
                self.layer2 = torch.nn.Linear(4, 1)

            def forward(self, input_):
                x = self.layer1(input_)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.layer2(x)
                return x

        rgr = Regressor()

        optimizer = torch.optim.Adam(
            [{"params": rgr.layer1.parameters(), "lr": 0},
             {"params": rgr.layer2.parameters(), "lr": 1e-3}]
        )

        X = torch.rand(124, 9)
        y = torch.rand(124, 1)
        train_data = torch.utils.data.TensorDataset(X, y)
        train_iter = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=False)

        X_test = torch.rand(73, 9)
        y_test = torch.rand(73, 1)

        loss_fn = torch.nn.MSELoss()

        dlu.train_test_utils.train(train_iter, model=rgr, loss_fn=loss_fn,
                                   optimizer=optimizer, num_epochs=20,
                                   test_data=(X_test, y_test),
                                   eval_test_R2=True)
    """
    typeguard.check_argument_types()

    if verbose:
        print('************* Training in progress *************')
        print("Training on:", device)
        print('')
    # END IF

    model = model.to(device)

    if test_data is not None:
        repackaged_test_data = unpack_repack_fn(test_data)
        X_test = repackaged_test_data.get_X(device)
        y_test = repackaged_test_data.get_y(device)
        options_test = repackaged_test_data.get_options(device)  # dict
    # END IF

    train_losses = []
    test_losses = []
    batch_counter = -1

    for epoch in range(num_epochs):
        if verbose_each_epoch and verbose_each_batch:
            print('Epoch %d:' % epoch)
        # END IF

        t0 = time.time()
        for training_data in train_iter:  # batch by batch
            batch_counter += 1

            repackaged_train_data = unpack_repack_fn(training_data)
            X = repackaged_train_data.get_X(device)
            y = repackaged_train_data.get_y(device)
            options_to_model = repackaged_train_data.get_options(device)  # dict

            y_pred = model(X, **options_to_model, **static_options_to_model)
            loss = loss_fn(y_pred, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append((batch_counter, loss.cpu().item()))

            if test_data is not None and eval_each_batch:
                y_pred_on_test = model(X_test, **options_test, **static_options_to_model)
                test_loss = loss_fn(y_pred_on_test, y_test)

                test_losses.append((batch_counter, test_loss.cpu().item()))

                if eval_test_accuracy:
                    test_accuracy = _calc_accuracy(y_test, y_pred_on_test)
                # END IF
                if eval_test_AUC:
                    test_AUC = _calc_AUC(y_test, y_pred_on_test)
                # END IF
                if eval_test_R2:
                    test_R2 = _calc_R2(y_test, y_pred_on_test)
                # END IF

                del y_pred_on_test  # free up memory
            # END IF

            if verbose_each_batch:
                train_loss_txt = 'Train loss: %.4f. ' % loss.item()
                if test_data is not None and eval_each_batch:
                    test_loss_txt = 'Test loss: %.4f. ' % test_loss.cpu().item()
                    if eval_test_accuracy:
                        test_accuracy_txt = 'Test accuracy: %.3f. ' % test_accuracy
                    else:
                        test_accuracy_txt = ''
                    # END IF-ELSE
                    if eval_test_AUC:
                        test_AUC_txt = 'Test AUC: %.3f. ' % test_AUC
                    else:
                        test_AUC_txt = ''
                    # END IF-ELSE
                    if eval_test_R2:
                        test_R2_txt = 'Test R2: %.3f. ' % test_R2
                    else:
                        test_R2_txt = ''
                    # END IF-ELSE
                else:
                    test_loss_txt = ''
                    test_accuracy_txt = ''
                    test_AUC_txt = ''
                    test_R2_txt = ''
                # END IF
                txt = train_loss_txt + test_loss_txt + test_accuracy_txt \
                    + test_AUC_txt + test_R2_txt
                print(txt)
            # END IF
        # END FOR

        dt = time.time() - t0
        if verbose_each_epoch:
            epoch_txt_ = '\nEpoch %d - ' % epoch
            train_loss_txt_ = 'Train loss: %.4f. ' % loss.cpu().item()

            if test_data is not None:
                if eval_each_batch:  # use the result of the most recent batch
                    test_loss_this_epoch = test_loss  # use the loss of last batch

                    if eval_test_accuracy:
                        test_accuracy_this_epoch = test_accuracy
                    # END IF
                    if eval_test_AUC:
                        test_AUC_this_epoch = test_AUC
                    # END IF
                    if eval_test_R2:
                        test_R2_this_epoch = test_R2
                    # END IF
                else:  # otherwise, evaluate right here right now
                    y_pred_on_test = model(X_test, **options_test, **static_options_to_model)
                    test_loss_this_epoch = loss_fn(y_pred_on_test, y_test)
                    test_losses.append((batch_counter, test_loss_this_epoch))

                    if eval_test_accuracy:
                        test_accuracy_this_epoch = _calc_accuracy(y_test, y_pred_on_test)
                    # END IF
                    if eval_test_AUC:
                        test_AUC_this_epoch = _calc_AUC(y_test, y_pred_on_test)
                    # END IF
                    if eval_test_R2:
                        test_R2_this_epoch = _calc_R2(y_test, y_pred_on_test)
                    # END IF

                    del y_pred_on_test  # free up memory
                # END IF

                test_loss_txt_ = 'Test loss: %.4f. ' % test_loss_this_epoch
                if eval_test_accuracy:
                    test_accuracy_txt_ = 'Test accuracy: %.3f. ' % test_accuracy_this_epoch
                else:
                    test_accuracy_txt_ = ''
                # END IF-ELSE
                if eval_test_AUC:
                    test_AUC_txt_ = 'Test AUC: %.3f. ' % test_AUC_this_epoch
                else:
                    test_AUC_txt_ = ''
                # END IF-ELSE
                if eval_test_R2:
                    test_R2_txt_ = 'Test R2: %.3f. ' % test_R2_this_epoch
                else:
                    test_R2_txt_ = ''
                # END IF-ELSE
            else:
                test_loss_txt_ = ''
                test_accuracy_txt_ = ''
                test_AUC_txt_ = ''
                test_R2_txt_ = ''
            # END IF

            time_txt_ = '(Time: %s)' % _seconds_to_hms(dt)
            txt_ = epoch_txt_ + train_loss_txt_ + test_loss_txt_ \
                 + test_accuracy_txt_ + test_AUC_txt_ + test_R2_txt_ + time_txt_
            print(txt_)
        # END IF
        if verbose_each_epoch and verbose_each_batch:
            print('----------------------------------')
        # END IF
    # END FOR

    if plot_learning_curve:
        plt.figure()
        plt.plot(*zip(*train_losses), marker='o', label='Train', alpha=0.8)
        if len(test_losses) > 0:
            plt.plot(*zip(*test_losses), marker='^', label='Test', alpha=0.8)
        # END IF
        plt.grid(ls=':', lw=0.5)
        plt.xlabel('Training step')
        plt.ylabel('Loss')
        plt.legend(loc='best')
    # END IF

#%%----------------------------------------------------------------------------
def _calc_accuracy(y_true: torch.Tensor, y_pred_raw: torch.Tensor) -> float:
    typeguard.check_argument_types()

    y_pred_class = torch.argmax(y_pred_raw, dim=1).cpu()
    accuracy = accuracy_score(
        y_true=y_true.cpu().detach().numpy(),
        y_pred=y_pred_class.detach().numpy()
    )

    typeguard.check_return_type(accuracy)
    return accuracy

#%%----------------------------------------------------------------------------
def _calc_AUC(y_true: torch.Tensor, y_pred_raw: torch.Tensor) -> float:
    typeguard.check_argument_types()

    AUC = roc_auc_score(
        y_true=y_true.cpu().detach().numpy(),
        y_score=y_pred_raw[:, 1].cpu().detach().numpy()
    )

    typeguard.check_return_type(AUC)
    return AUC

#%%----------------------------------------------------------------------------
def _calc_R2(y_true: torch.Tensor, y_pred_raw: torch.Tensor) -> float:
    typeguard.check_argument_types()

    R2 = r2_score(
        y_true=y_true.cpu().detach().numpy(),
        y_pred=y_pred_raw.cpu().detach().numpy()
    )

    typeguard.check_return_type(R2)
    return R2

#%%----------------------------------------------------------------------------
def _seconds_to_hms(seconds: float) -> str:
    if seconds <= 60:
        return '%.1f sec' % seconds
    # END IF

    if seconds <= 3600:
        return time.strftime('%M min, %s sec', time.gmtime(seconds))
    # END IF

    return time.strftime('%H hr, %M min, %S sec', time.gmtime(seconds))
