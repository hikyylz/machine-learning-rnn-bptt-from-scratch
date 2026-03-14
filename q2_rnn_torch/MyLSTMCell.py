import torch
from torch.nn import functional
from AbstractRNNCell import AbstractRNNCell

class MyLSTMCell(AbstractRNNCell):

    def __init__(self, input_size: int, hidden_size: int):
        """
        Create an LSTM operating on inputs of feature dimension 
        `input_size`, maintaining a hidden state of size `hidden_size`.
        """
        super().__init__()
        self.hidden_size = hidden_size
        #####Start Subtask 2a#####
        self.W_i = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_i = torch.nn.Parameter(torch.zeros(hidden_size))
        
        self.W_f = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_f = torch.nn.Parameter(torch.zeros(hidden_size))

        self.W_o = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_o = torch.nn.Parameter(torch.zeros(hidden_size))    

        self.W_g = torch.nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.empty((hidden_size, input_size + hidden_size))))
        self.bias_g = torch.nn.Parameter(torch.zeros(hidden_size))
        #####End Subtask 2a#####
        #no hidden state initially.
        self.has_hidden_state = False
        #unlike parameters, no gradients are computed for buffers
        #they can be accessed like other attributes of the model
        self.register_buffer(
            "hidden_state", torch.zeros((self.hidden_size, 1)))
        self.register_buffer("cell_state", torch.zeros((self.hidden_size, 1)))

    def get_rnn_type(self):
        return "LSTM"

    def forward(self, x: torch.Tensor, 
                reset_hidden_state: bool = True) -> torch.Tensor:  
        batch_sz, seq_len, feat_dim = x.shape

        if(reset_hidden_state or not self.has_hidden_state):
            self.hidden_state = torch.zeros(
                (self.hidden_size, batch_sz), device=self.W_f.device)
            self.cell_state = torch.zeros((self.hidden_size, batch_sz), 
                                          device=self.W_f.device)
            self.has_hidden_state = True
        
        outputs = []
        for t in range(seq_len):
            curr_x = x[:, t, :]
            #####Start Subtask 2b#####
            combined_input = torch.concat((curr_x.T, self.hidden_state), dim=0)
            #implement LSTM update equations
            current_i = torch.sigmoid(torch.matmul(
                self.W_i, combined_input) + self.bias_i[:, None])
            current_f = torch.sigmoid(torch.matmul(
                self.W_f, combined_input) + self.bias_f[:, None])
            current_o = torch.sigmoid(torch.matmul(
                self.W_o, combined_input) + self.bias_o[:, None])
            cell_state_tmp = torch.tanh(torch.matmul(
                self.W_g, combined_input) + self.bias_g[:, None])
            self.cell_state = (
                current_f*self.cell_state + current_i*cell_state_tmp)
            self.hidden_state = torch.tanh(self.cell_state)*current_o
            outputs.append(self.hidden_state.T)
            #####End Subtask 2b#####
        outputs = torch.stack(outputs, dim=1)
        assert outputs.shape == (batch_sz, seq_len, self.hidden_size)            

        return outputs
