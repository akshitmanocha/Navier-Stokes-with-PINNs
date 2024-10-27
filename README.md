# Navier-Stokes-with-PINNs
In this we solved 2-Dimentional Navier Stokes equation with the help of Physics informed Neural Networks

## About the code 

### Model.py
'''python
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2)
        )
    def forward(self, x, y, t):
        return self.net(torch.hstack((x, y, t)))
        '''
We used 8 hidden layers along with tanh activation function as the data is complex
        
