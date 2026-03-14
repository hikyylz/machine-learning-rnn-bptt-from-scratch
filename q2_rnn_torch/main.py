import torch
from SineDataset import SineDataset
from CharSequenceDataset import CharSequenceDataset
from BasicRNNCell import BasicRNNCell
from MyLSTMCell import MyLSTMCell
from RNNPlusLinearLayer import RNNPlusLinearLayer
from loops import train_loop, test_loop_sines, test_loop_mem_task
from memory_task import memory_task_loss
import argparse

def make_optimizer(lr):
    """
    Returns a function that initializes an optimizer on a set of parameters.
    The optimizer will have learning rate lr.
    """
    return (lambda params: torch.optim.Adam(params, lr))

if __name__ == "__main__":
    torch.random.manual_seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=2, help="Task number")
    parser.add_argument("--num_epochs", type=int, default=400, 
        help="Number of training epochs")
    parser.add_argument("--batch_sz", type=int, default=500, 
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--rnn_sz", type=int, default=128, help="RNN/LSTM size")
    parser.add_argument("--rnn_type", type=str, default="BasicRNN", 
                        help="RNN type: BasicRNN or LSTM")
    parsed, unparsed = parser.parse_known_args()

    if(parsed.task == 1):
        train_dataset_size = 10000
        test_dataset_size = 5
        samples_per_sequence = 100
        sample_step = 0.15
        #needed for test sequence generation to provide the RNN with some
        #"context" for future prediction
        num_bootstrapping_test_samples = 30

        train_dataset = SineDataset(
            train_dataset_size, samples_per_sequence, sample_step)
        test_dataset = SineDataset(
            test_dataset_size, 
            samples_per_sequence + num_bootstrapping_test_samples, sample_step)
        #input size = output size = 1 since the RNN operates on real numbers
        if(parsed.rnn_type == "BasicRNN"):
            cell = BasicRNNCell(
                input_size=1, hidden_size=parsed.rnn_sz)
        elif(parsed.rnn_type == "LSTM"):
            cell = MyLSTMCell(
                input_size=1, hidden_size=parsed.rnn_sz)
        else:
            print("Unknown type of RNN cell")
            exit(-1)
        #use mean square error as the loss function
        model = RNNPlusLinearLayer(cell, parsed.rnn_sz, 1)
        trained_model = train_loop(parsed.num_epochs, model, make_optimizer(
            parsed.lr), torch.nn.MSELoss(), parsed.batch_sz, train_dataset)
        test_loop_sines(
            trained_model, test_dataset, num_bootstrapping_test_samples)
    elif(parsed.task == 2):
        train_dataset_size = 10000
        test_dataset_size = 1000

        to_remember_len = 8
        blank_separation_len = 5
        #8 possible characters to remember + blank + delimiter = 10
        num_possible_chars = 10 

        train_dataset = CharSequenceDataset(
            train_dataset_size, to_remember_len, blank_separation_len)
        test_dataset = CharSequenceDataset(test_dataset_size, to_remember_len,
                                           blank_separation_len)
        
        #input size = output size = num_possible_chars since we're using 
        # one-hot encoding
        if(parsed.rnn_type == "BasicRNN"):
            cell = BasicRNNCell(input_size=num_possible_chars, 
                                hidden_size=parsed.rnn_sz)
        elif(parsed.rnn_type == "LSTM"):
            cell = MyLSTMCell(
                input_size=num_possible_chars, hidden_size=parsed.rnn_sz)
        else:
            print("Unknown type of RNN cell")
            exit(-1)
        model = RNNPlusLinearLayer(cell, parsed.rnn_sz, num_possible_chars)
        trained_model = train_loop(parsed.num_epochs, model, make_optimizer(
            parsed.lr), lambda y, gt: memory_task_loss(y, gt, to_remember_len), 
            parsed.batch_sz, train_dataset)
        test_loop_mem_task(trained_model, test_dataset, parsed.batch_sz, 
                           to_remember_len)            

    else:
        print("Unknown task number")
        exit(-1)

