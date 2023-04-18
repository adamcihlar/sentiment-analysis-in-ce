import torch.nn.functional as F
import torch


def corn_loss_weighted(logits, y_train, num_classes, weights):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.
    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.
    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    weights: torch.tensor, shape(num_classes)
        Weights of individual classes.
    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.
    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> y_train = torch.tensor([0, 0, 1, 1, 1,1,1])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    weights = torch.tensor(y_train.shape[0]/pd.Series(y_train).value_counts().sort_index())
    weights = F.normalize(weights, dim=0)
    weights = weights/2
    weights * pd.Series(y_train).value_counts().sort_index()
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    sets = []
    for i in range(num_classes - 1):
        label_mask = y_train > i - 1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.0
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(
            ((F.logsigmoid(pred) * train_labels) * weights[task_index])
            + (
                (F.logsigmoid(pred) - pred)
                * (1 - train_labels)
                * weights[task_index + 1]
            )
        )
        losses += loss

    return losses / num_examples
