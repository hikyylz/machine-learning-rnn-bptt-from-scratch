from AbstractRNNCell import AbstractRNNCell


import torch
from einops import rearrange
from torch import Tensor
from torch.nn import Module


class RNNPlusLinearLayer(Module):

    def __init__(self, rnn_cell: AbstractRNNCell,
                 hidden_size: int, output_size: int):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.W_hy = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((output_size, hidden_size))))
        self.bias_hy = torch.nn.Parameter(torch.zeros(output_size))
        self.output_size = output_size

    def forward(self, x: Tensor, reset_hidden_state: bool = True) -> Tensor:
        """
        Performs a forward pass of the RNN and the output linear layer 
        on input x.
        Args:
            x: a tensor of one of the following shapes:
                1) [`B`, `L`, `D`], where `B` is the batch 
                size (number of sequences processed in parallel), `L` is 
                the length of the sequences and `D` is the feature 
                dimensionality of a single sample of the sequence.
                2) [`B`, `L`], in which case everything is as above 
                except for `D` being inferred as 1
                3) [`B`], in which case both `L, D` are inferred as 1
            reset_hidden_state: whether to start processing `x` from
                scratch or continue from the last value of the hidden 
                state of the RNN. Useful for splitting a sequence and 
                processing in more than one calls.
        Returns:
            A tensor containing the final `output_size`-dimensional output 
            for every sample of every sequence of the batch `x` (the
            other dimensions match these of `x`).
        """
        if(len(x.shape) == 1):
            x = x[:, None, None] #assume length and feat dim 1
        elif(len(x.shape) == 2):
            x = x[:, :, None] #assume feat dim 1            
        rnn_res = self.rnn_cell(x, reset_hidden_state)
        batch_sz, seq_len, feat_dim = rnn_res.shape
        #we create an "artificial batch" for the purposes of matmul
        #rnn_res = torch.reshape(rnn_res, (batch_sz*seq_len, feat_dim)).T
        rnn_res = rearrange(rnn_res, "B S D -> D (B S)")
        res = (torch.matmul(self.W_hy, rnn_res) + self.bias_hy[:, None])
        #undo the artificial batching
        res = rearrange(res, "D (B S) -> B S D", B=batch_sz)
        #res = torch.reshape(res.T, (batch_sz, seq_len, -1))
        assert res.shape == (x.shape[0], x.shape[1], self.output_size)
        return torch.squeeze(res)