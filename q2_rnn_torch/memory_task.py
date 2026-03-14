import torch
def memory_task_loss(outputs : torch.tensor, 
                     ground_truth : torch.tensor,
                     to_remember_len: int) -> torch.tensor:
    """
    Computes an appropriate classification loss for every character of 
    the predicted sequence.
    Args:
        outputs: a tensor of shape [`B`, `L`, `C`], `B` being the batch 
            size, `L` being the sequence length and `C` being the 
            number of classes. The last dimension should contain the
            unnormalized output of the "RNN + linear layer" model for
            each sample.
        ground_truth: a tensor of shape [`B`, `L`] containing the class indices
            for every sample of every sequence as a number in {0, ..., `C`-1}
        to_remember_len: the length of the sequence that has to be 
            reproduced. Necessary for applying the loss to the current 
            outputs
    Returns:
        The average classification loss for all samples
    """
    #####Start Subtask 2e#####
    outputs = torch.permute(outputs[:, -to_remember_len:, :], (0, 2, 1))
    return torch.nn.CrossEntropyLoss()(outputs, 
                                       ground_truth[:, -to_remember_len:])
    #####End Subtask 2e#####

def memory_task_acc(outputs: torch.Tensor, ground_truth: torch.Tensor, 
                    to_remember_len: int) -> torch.Tensor:
    """
    Computes the accuracy of the predicted output vs the ground truth,
    only for the last `to_remember_len` characters of every sequence.
    Args:
        outputs: a tensor of shape [`B`, `L`, `C`], `B` being the batch 
            size, `L` being the sequence length and `C` being the 
            number of classes. The last dimension should contain the
            unnormalized output of the "RNN + linear layer" model for
            each sample.
        ground_truth: a tensor of shape [`B`, `L`] containing the ground
            truth indices for every sample of every sequence as a number 
            in {0, ..., `C`-1}.
    Returns:
        The average accuracy of the prediction for the last 
        `to_remember_len` characters of every sequence (averaged over 
        characters and sequences).
    """
    #####Start Subtask 2f#####
    outputs = torch.softmax(outputs, dim=2)
    outputs = torch.argmax(outputs, dim=2)
    acc = torch.mean((outputs[:, -to_remember_len:] == ground_truth[
        :, -to_remember_len:]).float())
    return acc
    #####End Subtask 2f#####