import torch
from torch.nn import functional
from AbstractRNNCell import AbstractRNNCell

class BasicRNNCell(AbstractRNNCell):

    def __init__(self, input_size: int, hidden_size: int):
        """
        Create a basic RNN operating on inputs of feature dimension 
        `input_size`, maintaining a hidden state of dimension `hidden_size`
        (this is also the dimension of the output at each step).
        """
        super().__init__()
        self.hidden_size = hidden_size
        #initialize the weight matrix (with weights for both input and
        #hidden state) as a trainable parameter
        self.W_xh_hh = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        #same for bias vector
        self.bias_hh = torch.nn.Parameter(torch.zeros(hidden_size))
        #no hidden state initially.
        self.has_hidden_state = False
        #unlike parameters, no gradients are computed for buffers
        #they can be accessed like other attributes of the model
        self.register_buffer(
            "hidden_state", torch.zeros((self.hidden_size, 1)))

    def get_rnn_type(self) -> str:
        return "BasicRNN"

    def forward(self, x: torch.Tensor, 
                reset_hidden_state: bool = True)-> torch.Tensor:
        batch_sz, seq_len, feat_dim = x.shape
        #initialize hidden state to zeros if it's the first time processing
        #a batch or if we want the RNN to start from scratch
        if(reset_hidden_state or not self.has_hidden_state):
            #on the same device as our params
            self.hidden_state = torch.zeros(
                (self.hidden_size, batch_sz), device=self.W_xh_hh.device)
            self.has_hidden_state = True

        outputs = []
        for t in range(seq_len):
            curr_x = x[:, t, :] #select the t-th timestep of all sequences

            combined_input = torch.concat((curr_x.T, self.hidden_state), dim=0)
            #implement RNN update equation
            self.hidden_state = (torch.matmul(self.W_xh_hh, combined_input) + 
                                 self.bias_hh[:, None])
            self.hidden_state = functional.relu(self.hidden_state)
            outputs.append(self.hidden_state.T)
        outputs = torch.stack(outputs, dim=1)
        assert outputs.shape == (batch_sz, seq_len, self.hidden_size)

        return outputs
