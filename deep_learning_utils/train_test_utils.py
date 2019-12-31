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
        test_iter: torch.utils.data.DataLoader = None,
        verbose: bool = True,
        verbose_each_batch: bool = True,
        verbose_each_epoch: bool = True,
        eval_on_CPU: bool = False,
        eval_each_batch: bool = False,
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
        :py:class:`~deep_learning_utils.data_util_classes.FeatureLabelOptionPack`
        object. An example function is :py:meth:`~unpack_repack_data`.
    static_options_to_model : Dict[str, Any]
        "Static" keyword arguments to pass to the ``model``'s ``forward()``
        method. "Static" means that these keyword arguments don't change in
        different mini batches or different epochs. You need to pack them into
        a dictionary, whose keys are the argument names and values are argument
        values. It can be an empty dict, which means that no additional
        arguments will be passed to ``forward()``.
    test_iter : torch.utils.data.DataLoader or ``None``
        Test data to evaluate the model performance on. If you do not want to
        evaluate during training, you can pass in ``None``.
    verbose : bool
        Whether to show messages of progress on the console. Default = True.
    verbose_each_batch : bool
        Whether to print loss values at each batch. Default = True.
    verbose_each_epoch : bool
        Whether to print loss values at each epoch. Default = True.
    eval_on_CPU : bool
        Whether to perform model evauation (using ``test_iter``) on the CPU.
        Set this to ``True`` if your GPU's memory is really limited.
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
        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_iter = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        loss_fn = torch.nn.CrossEntropyLoss()

        dlu.train_test_utils.train(train_iter=train_iter, model=clf, loss_fn=loss_fn,
                                   optimizer=optimizer, num_epochs=20,
                                   test_iter=test_iter,
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
        test_data = torch.utils.data.TensorDataset(X_test, y_test)
        test_iter = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

        loss_fn = torch.nn.MSELoss()

        dlu.train_test_utils.train(train_iter=train_iter, model=rgr, loss_fn=loss_fn,
                                   optimizer=optimizer, num_epochs=20,
                                   test_iter=test_iter,
                                   eval_test_R2=True)

    """
    typeguard.check_argument_types()

    if verbose:
        print('************* Training in progress *************')
        print("Training on:", device)
        print('')
    # END IF

    model = model.to(device)

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

            if test_iter is not None and eval_each_batch:
                test_scores, eval_txt = eval_model(
                    model=model,
                    test_iter=test_iter,
                    loss_fn=loss_fn,
                    unpack_repack_fn=unpack_repack_fn,
                    static_options_to_model=static_options_to_model,
                    training_device=device,
                    eval_on_CPU=eval_on_CPU,
                    eval_accuracy=eval_test_accuracy,
                    eval_AUC=eval_test_AUC,
                    eval_R2=eval_test_R2,
                    verbose=False,
                )
                test_losses.append((batch_counter, test_scores['loss']))
            else:
                eval_txt = ''
            # END IF-ELSE

            if verbose_each_batch:
                train_loss_txt = 'Train loss: %.4f. ' % loss.item()
                txt = train_loss_txt + eval_txt
                print(txt)
            # END IF
        # END FOR

        dt = time.time() - t0
        if verbose_each_epoch:
            epoch_txt_ = '\nEpoch %d - ' % epoch
            train_loss_txt_ = 'Train loss: %.4f. ' % loss.cpu().item()

            if test_iter is not None:
                if eval_each_batch and verbose_each_batch:
                    test_loss_this_epoch = test_scores['loss']
                    eval_txt_this_epoch = eval_txt
                else:  # otherwise, evaluate right here
                    test_scores_this_epoch, eval_txt_this_epoch = eval_model(
                        model=model,
                        test_iter=test_iter,
                        loss_fn=loss_fn,
                        unpack_repack_fn=unpack_repack_fn,
                        static_options_to_model=static_options_to_model,
                        training_device=device,
                        eval_on_CPU=eval_on_CPU,
                        eval_accuracy=eval_test_accuracy,
                        eval_AUC=eval_test_AUC,
                        eval_R2=eval_test_R2,
                        verbose=False,
                    )
                    test_loss_this_epoch = test_scores_this_epoch['loss']
                # END IF
            else:
                eval_txt_this_epoch = ''
            # END IF

            test_losses.append((batch_counter, test_loss_this_epoch))

            time_txt_ = '(Time: %s)' % _seconds_to_hms(dt)
            txt_ = epoch_txt_ + train_loss_txt_ + eval_txt_this_epoch + time_txt_
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
        plt.show()
    # END IF

#%%----------------------------------------------------------------------------
def eval_model(
        *,
        model: torch.nn.Module,
        test_iter: torch.utils.data.DataLoader,
        loss_fn: torch.nn.modules.loss._Loss,
        unpack_repack_fn: Callable[[Any], FeatureLabelOptionPack],
        static_options_to_model: Dict[str, Any],
        training_device: Union[str, torch.device],
        eval_on_CPU: bool,
        eval_accuracy: bool,
        eval_AUC: bool,
        eval_R2: bool,
        verbose: bool,
) -> Tuple[Dict[str, Union[float, None]], str]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : torch.nn.Module
        Model object.
    test_iter : torch.utils.data.DataLoader
        Test data iteration.
    loss_fn : torch.nn.modules.loss._Loss
        Loss function.
    unpack_repack_fn : Callable[[Any], FeatureLabelOptionPack]
        A function that unpacks each item in ``train_iter`` (or unpacks
        ``test_data``) and repackages the information into a
        :py:class:`~deep_learning_utils.data_util_classes.FeatureLabelOptionPack`
        object. An example function is :py:meth:`~unpack_repack_data`.
    static_options_to_model : Dict[str, Any]
        "Static" keyword arguments to pass to the ``model``'s ``forward()``
        method. "Static" means that these keyword arguments don't change in
        different mini batches or different epochs. You need to pack them into
        a dictionary, whose keys are the argument names and values are argument
        values. It can be an empty dict, which means that no additional
        arguments will be passed to ``forward()``.
    training_device : Union[str, torch.device]
        The device to train the model. This parameter is needed because we
        often need to keep training the model after evaluation, so we need to
        send the model back to training device.
    eval_on_CPU : bool
        Whether to perform model evauation (using ``test_iter``) on the CPU.
        Set this to ``True`` if your GPU's memory is really limited.
    eval_accuracy : bool
        Whether to evaluate accuracy. Suitable for classification problem.
    eval_AUC : bool
        Whether to evaluate AUC. Suitable for binary classification only.
    eval_R2 : bool
        Whether to evaluate R2 (coefficient of determination). Suitable for
        regression problem only.
    verbose : bool
        Whether to show the loss/scores on the console.

    Returns
    -------
    test_scores : Dict[str, Union[float, None]]
        The test scores. A dictionary of keys "loss", "accuracy", "AUC", and
        "R2". The values are either floats or None.
    eval_txt : str
        The text that looks like "Test loss: ***. Test accuracy: ***".
    """
    typeguard.check_argument_types()

    eval_device = "cpu" if eval_on_CPU else training_device

    y_pred_on_test_list = []
    y_test_list = []

    for test_data in test_iter:
        repackaged_test_data = unpack_repack_fn(test_data)
        X_test_this_iter = repackaged_test_data.get_X(eval_device)
        y_test_this_iter = repackaged_test_data.get_y(eval_device)
        options_to_model = repackaged_test_data.get_options(eval_device)

        model.to(eval_device)  # in-place
        y_pred_on_test_this_iter = model(
            X_test_this_iter,
            **options_to_model,
            **static_options_to_model
        ).detach()  # detach to save RAM

        y_pred_on_test_list.append(y_pred_on_test_this_iter)
        y_test_list.append(y_test_this_iter)
    # END FOR

    y_pred_on_test = torch.cat(y_pred_on_test_list)
    y_test = torch.cat(y_test_list)

    test_loss = loss_fn(y_pred_on_test, y_test).cpu().item()

    acc = _calc_accuracy(y_test, y_pred_on_test) if eval_accuracy else None
    AUC = _calc_AUC(y_test, y_pred_on_test) if eval_AUC else None
    R2 = _calc_R2(y_test, y_pred_on_test) if eval_R2 else None

    del y_pred_on_test  # save RAM

    loss_txt = 'Test loss: %.4f. ' % test_loss
    acc_txt = 'Test accuracy: %.3f. ' % acc if eval_accuracy else ''
    AUC_txt = 'Test AUC: %.3f. ' % AUC if eval_AUC else ''
    R2_txt = 'Test R2: %.3f. ' % R2 if eval_R2 else ''
    eval_txt = loss_txt + acc_txt + AUC_txt + R2_txt

    if verbose:
        print(eval_txt)
    # END IF-ELSE

    model.to(training_device)  # move back to the device for training

    test_scores = {'loss': test_loss,
                   'accuracy': acc,
                   'AUC': AUC,
                   'R2': R2}

    return test_scores, eval_txt

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
        return time.strftime('%M min, %S sec', time.gmtime(seconds))
    # END IF

    return time.strftime('%H hr, %M min, %S sec', time.gmtime(seconds))
