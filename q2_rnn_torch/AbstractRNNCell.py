from torch.nn import Module
from torch import Tensor
from abc import ABC, abstractmethod


class AbstractRNNCell(Module, ABC):
    """
    An abstract base class for our simple recurrent networks.
    """
    @abstractmethod
    def get_rnn_type(self) -> str:
        """
        Return the type of RNN being implemented as a string.
        """
        pass

    @abstractmethod
    def forward(self, x: Tensor, reset_hidden_state: bool = True) -> Tensor:
        """
        Performs a forward pass of the RNN on input x.
        Args:
            x: a tensor of shape [`B`, `L`, `D`], where `B` is the batch 
                size (number of sequences processed in parallel), `L` is 
                the length of the sequences and `D` is the feature 
                dimensionality of a single sample of the sequence.
            reset_hidden_state: whether to start processing `x` from
                scratch or continue from the last value of the hidden 
                state of the RNN. Useful for splitting a sequence and 
                processing in more than one calls.
        Returns:
            A tensor of shape [`B`, `L`, `D'`] containing the RNN unit's 
            `D'`-dimensional output for every sample of every sequence 
            of the batch.
        """
        pass

