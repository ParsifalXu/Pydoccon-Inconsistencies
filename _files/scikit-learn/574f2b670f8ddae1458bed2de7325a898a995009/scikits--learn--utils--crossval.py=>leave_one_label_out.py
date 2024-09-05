def leave_one_label_out(label):
    """
    Leave-One-Label_Out cross validation:
    Provides train/test indexes to split data in train test sets

    Parameters
    ----------
    label : list
            List of labels

    Examples
    ----------
    >>> import numpy as np
    >>> from scikits.learn.utils import crossval
    >>> X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    >>> y = [1, 2, 1, 2]
    >>> label = [1,1,2,2]
    >>> lol = crossval.leave_one_label_out(label)
    >>> for train_index, test_index in lol:
    ...    print "TRAIN:", train_index, "TEST:", test_index
    ...    X_train, X_test, y_train, y_test = crossval.split(train_index,            test_index, X, y)
    ...    print X_train, X_test, y_train, y_test
    TRAIN: [False False  True  True] TEST: [ True  True False False]
    [[5 6]
     [7 8]] [[1 2]
     [3 4]] [1 2] [1 2]
    TRAIN: [ True  True False False] TEST: [False False  True  True]
    [[1 2]
     [3 4]] [[5 6]
     [7 8]] [1 2] [1 2]

    """
    for i in np.unique(label):
        test_index = np.zeros(len(label), dtype=np.bool)
        test_index[label == i] = True
        train_index = np.logical_not(test_index)
        yield (train_index, test_index)