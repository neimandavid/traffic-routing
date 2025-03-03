import torch
from torch import nn

class Net(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            #Input: Assume 4 roads, 3 lanes each, store #stopped and #total on each. Also store current phase and duration, and really hope the phases and roads are in the same order
            #So 26 inputs. Phase, duration, L1astopped, L1atotal, ..., L4dstopped, L4dtotal

            nn.Linear(in_size, hidden_size), #Blindly copying an architecture
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.LeakyReLU(),
            nn.Linear(hidden_size, out_size)
        )
        
    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))