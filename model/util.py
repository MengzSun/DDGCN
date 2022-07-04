import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "model_saved/ckpt_nn.model"

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('training process',self.counter)
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print('training process:',self.counter)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                    'Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min,val_loss))
        # torch.save(model.state_dict(), self.model_path)
        #============python3.6 与 torch_geometric 不能用torch.save了
        # torch.save(model, self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.model_path)

class EarlyStopping_acc:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "model_saved/ckpt_nn.model"

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('training process',self.counter)
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print('training process:',self.counter)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                    'Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min,val_loss))
        # torch.save(model.state_dict(), self.model_path)
        #============python3.6 与 torch_geometric 不能用torch.save了
        # torch.save(model, self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.model_path)

class EarlyStopping_early:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "model_saved/ckpt_nn.model"

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print('training process',self.counter)
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print('training process:',self.counter)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                    'Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min,val_loss))
        # torch.save(model.state_dict(), self.model_path)
        #============python3.6 与 torch_geometric 不能用torch.save了
        torch.save(model, self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.model_path)

def evaluation(outputs,labels):
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    correct = torch.sum(torch.eq(outputs,labels)).item()
    return correct

def macro_f1(pred, targ, num_classes=None):
    # pred = torch.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()  # 预测为i，且标签的确为i的
        fp = ((pred == i) & (targ != i)).sum().item()  # 预测为i，但标签不是为i的
        fn = ((pred != i) & (targ == i)).sum().item()  # 预测不是i，但标签是i的
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision_real = precision[0]
    precision_fake = precision[1]
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall_real = recall[0]
    recall_fake = recall[1]
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real)
    f1_fake = 2 * (precision_fake * recall_fake) / (precision_fake + recall_fake)
    return f1, precision, recall, f1_real, precision_real, recall_real, f1_fake, precision_fake, recall_fake


def accuracy(pred, targ):
    # pred = torch.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc