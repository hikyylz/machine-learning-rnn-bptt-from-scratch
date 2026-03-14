import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from typing import Tuple

class CharSequenceDataset(Dataset):
    def __init__(self, dataset_size: int, to_remember_len: int, 
                 blank_separation_len: int):
        """
        Create a dataset containing character sequences of length
        `to_remember_len` random characters from {0, ..., 7}, followed 
        by `blank_separation_len` blanks ("8"), a delimiter ("9") and 
        another `to_remember_len` blanks. The ground truth for each 
        sequence is a sequence of 
        `to_remember_len + blank_separation + 1` blanks followed by the
        `to_remember_len` characters of the sequence.
        """
        #define the integers corresponding to the blank and delimiter chars
        blank_char = 8
        delim_char = 9

        self.dataset_size = dataset_size
        #generate random sequences, blanks and delimiters
        to_remember_seq = torch.randint(
            0, blank_char, (dataset_size, to_remember_len))
        blanks = torch.full((dataset_size, blank_separation_len),
                            fill_value=blank_char)
        delimiters = torch.full((dataset_size, 1), fill_value=delim_char)
        #generate the "space" for the NN's output
        blanks_for_answer = torch.full(
            (dataset_size, to_remember_len), fill_value=blank_char)
        #concatenate everything and store
        sequences = torch.concat((to_remember_seq, blanks, delimiters,
                                  blanks_for_answer), dim=1).long()
        self.sequences = one_hot(sequences)
        assert self.sequences.shape == (
            self.dataset_size, 
            2*to_remember_len + blank_separation_len + 1, 10)
        #the ground truth is simply a number of blanks and then the initial
        #sentence
        blanks = torch.full((dataset_size, to_remember_len + 
                             blank_separation_len + 1), fill_value=blank_char)
        self.ground_truth = torch.concat((blanks, to_remember_seq), 
                                         dim=1).long()
        assert self.ground_truth.shape == (
            self.dataset_size, 2*to_remember_len + blank_separation_len + 1)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the i-th character sequence and corresponding ground
        truth as a tuple. The input sequence is one-hot encoded, and
        as such is of shape (`L`, 9), where `L` is the total sequence
        length. The ground truth is not one-hot encoded, i.e. is of
        shape (`L`,).
        """
        return (self.sequences[i, :], self.ground_truth[i, :])