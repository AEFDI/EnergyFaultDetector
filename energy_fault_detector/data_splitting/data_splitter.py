
from typing import Tuple, Optional
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger('energy_fault_detector')


class BlockDataSplitter:
    """Split data into training and validation blocks. Train and val blocks will be chosen iteratively. Their
    size depends on train_block_size and val_block_size.

    Args:
        train_block_size: determines the size of one training block
        val_block_size: determines the size of a validation block

    Attributes:
        train_selection: array containing booleans which indicate which sample of the input x belongs to the training
            blocks.
        val_selection: array containing booleans which indicate which sample of the input x belongs to the validation
            blocks.

    Configuration example:

    .. code-block:: yaml

        train:
          data_splitter:
            train_block_size: 5040
            val_block_size: 1680

    """

    def __init__(self, train_block_size: int = 5040, val_block_size: int = 1680):
        train_block_size = 5040 if train_block_size is None else train_block_size
        val_block_size = 1680 if val_block_size is None else val_block_size
        self.train_block_size: int = train_block_size
        self.full_block_size: int = self.train_block_size + val_block_size

        self.train_selection: np.array = np.empty(0)
        self.val_selection: np.array = np.empty(0)

    def split(self, x: np.array, y: Optional[np.array] = None) -> Tuple:
        """
        This function constructs an array of booleans that indicate which samples belongs to the training data.
        At first the number of full blocks (training block + validation block) that fits into the data is determined.
        Each full block will then be divided into a training block and a validation block.
        If there is a remaining part at the end of data where no full block would fit, the remainder is defined as
        training data.
        After all training data was found, the validation data is defined as everything that is not training data.

        Args:
            x: data to split.
            y: data labels - optional
        """
        if self.full_block_size == self.train_block_size:
            logger.info('DataSplitter: Validation block size = 0, so no data split')
            if y is not None:
                return x, None, y, None
            return x, None

        # determine number of full blocks (train + val block) in data
        num_data = len(x)
        num_full_blocks = int(num_data / self.full_block_size)

        # split full blocks
        train_selection = pd.DataFrame([False] * num_data)
        for i in range(num_full_blocks):
            left = i * self.full_block_size
            right = left + self.train_block_size
            train_selection[left: right] = True

        # if there is a remainder, use it as training data
        left = num_full_blocks * self.full_block_size
        train_selection[left:] = True

        # cast from pandas dataframe to numpy array
        self.train_selection = train_selection[0].to_numpy()

        # validation data is everything that is not train data
        self.val_selection = ~self.train_selection

        train_data = x[self.train_selection]
        val_data = x[self.val_selection]
        if y is not None:
            train_normal_index = y[self.train_selection]
            val_normal_index = y[self.val_selection]
            return train_data, val_data, train_normal_index, val_normal_index

        return train_data, val_data
