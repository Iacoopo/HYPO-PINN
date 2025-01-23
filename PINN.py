import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time
import random
from tqdm import tqdm
from torchinfo import summary
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skfmm
from torchviz import make_dot

torch.manual_seed(4321)    
np.random.seed(4321) 
device = torch.device(1 if torch.cuda.is_available() else 'cpu') 
print(f"Device: {device}")

def positional_encoding(coords, L=10, include_input=True, log_scale=True):
    """
    coords: shape (N, 3) => x, y, z
    L: number of frequency octaves per coordinate
    include_input: whether to concatenate the original (x,y,z)
    log_scale: whether frequencies scale in powers of two, i.e. [1, 2, 4, 8, ...].
               If False, you might use a linear range or something else.

    Return:
       encoded: shape (N, d), where d = 3*(2*L + 1) if include_input else 3*(2*L).
    """
    # coords is (N, 3)
    out = []
    if include_input:
        # optionally include the raw x,y,z
        out.append(coords)

    # frequency schedule
    if log_scale:
        # typical approach: frequencies = [1, 2, 4, 8, ...]
        frequencies = 2.0 ** torch.linspace(0, L-1, L, device=coords.device)
    else:
        # linear frequencies, e.g. [1, 2, 3, ... L]
        frequencies = torch.linspace(1, L, L, device=coords.device)

    # For each coordinate dimension
    for i in range(3):
        # shape (N,1)
        x_i = coords[:, i:i+1]
        for f in frequencies:
            out.append(torch.sin(f * np.pi * x_i))
            out.append(torch.cos(f * np.pi * x_i))

    # Concatenate along dim=1
    encoded = torch.cat(out, dim=1)
    return encoded



class PINN(nn.Module):
    def __init__(self, L=10, init='kaiming_normal', include_input=True):
        super().__init__()
        self.L = L
        self.include_input = include_input
        # compute size of input after encoding
        if self.include_input:
            self.in_feats = 3 + 2 * L * 3  # 3 raw coords + (2L for each of x,y,z)
        else:
            self.in_feats = 2 * L * 3
        
        self.linears = nn.Sequential(
            nn.Linear(self.in_feats, 16),
            nn.Mish(),
            nn.Linear(16,16),
            nn.Mish(),
            nn.Linear(16,32),
            nn.Mish(),
            nn.Linear(32,64),
            nn.Mish(),
            nn.Linear(64,128),
            nn.Mish(),
            nn.Linear(128,256),
            nn.Mish(),
            nn.Linear(256,256),
            nn.Mish(),
            nn.Linear(256,128),
            nn.Mish(),
            nn.Linear(128,64),
            nn.Mish(),
            nn.Linear(64,32),
            nn.Mish(),
            nn.Linear(32,16),
            nn.Mish(),
            nn.Linear(16,16),
            nn.Mish(),
            nn.Linear(16, 1),
        )
        
        self.model_initialization(init)

    def model_initialization(self, init='kaiming_normal'):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif init == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif init == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0) 
                elif init == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
                else:
                    raise NotImplementedError("Initialization not implemented.")

    def forward(self, x):
        # x.shape = (batch_size, 3)
        x_enc = positional_encoding(x, L=self.L, include_input=self.include_input)
        return self.linears(x_enc)



class LossEikonal(nn.Module):
    """
    Create a loss function for Eikonal equation 
    physics informed neural network (PINN).
    """
    def __init__(self, w_eqn=1.0, w_data=1.0, w_heav=1.0):
        super().__init__()
        self.loss = nn.MSELoss()
        # Store the weights
        self.w_eqn = w_eqn
        self.w_data = w_data
        self.w_heav = w_heav

    def forward(self, model, params):
        vel = params['vel']
        inputs = params['input']
        data_pts = params['data_pts']
        data = params['data']

        # compute derivative of tau
        tau = model(inputs)
        tau_data = model(data_pts)

        g = ag.grad(outputs=tau, inputs=inputs, 
                    grad_outputs=torch.ones_like(tau), create_graph=True)[0]

        dtau_dx = g[..., 0].reshape(-1, 1)
        dtau_dy = g[..., 1].reshape(-1, 1)
        dtau_dz = g[..., 2].reshape(-1, 1)

        # Compute Eikonal (PDE) loss
        loss_eqn = self.loss(dtau_dx**2 + dtau_dy**2 + dtau_dz**2, 1.0 / (vel**2))

        # Heaviside/constraint term
        #  (1 - sign(tau)) * abs(tau) tries to enforce tau >= 0 near the source
        loss_heav = self.loss((1 - torch.sign(tau)) * torch.abs(tau), torch.zeros_like(tau))

        # Data (receiver) loss
        loss_data = self.loss(tau_data, data)

        # Weighted total loss
        loss_total = self.w_eqn * loss_eqn + self.w_data * loss_data + self.w_heav * loss_heav
        return loss_total, loss_eqn, loss_data


# Define the dimensions of the 3D model
nx, ny, nz = 100, 100, 100  # Grid size (X, Y, Z)
xmin, xmax = 0, 10  # X limits
ymin, ymax = 0, 10  # Y limits
zmin, zmax = 0, 10  # Z limits
x = np.linspace(xmin, xmax, nx)
y = np.linspace(ymin, ymax, ny)
z = np.linspace(zmin, zmax, nz)
delta_x = x[1] - x[0]
delta_y = y[1] - y[0]
delta_z = z[1] - z[0]

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Define velocity variations
# Base velocity with sinusoidal layering
velocity_base = 1500 + 300 * np.sin(2 * np.pi * Z / np.max(Z) + np.pi * X / np.max(X))

# Add Gaussian inclusions (e.g., pockets)
gaussian_1 = 500 * np.exp(-((X - 5) ** 2 + (Y - 5) ** 2 + (Z - 2) ** 2))
gaussian_2 = -400 * np.exp(-((X - 7) ** 2 + (Y - 3) ** 2 + (Z - 7) ** 2))
gaussian_3 = 300 * np.exp(-((X - 3) ** 2 + (Y - 8) ** 2 + (Z - 5) ** 2))

# Linearly increasing velocity with depth
velocity_depth = 200 * Z / np.max(Z)

# Combine features
velocity_model_3d = velocity_base + gaussian_1 + gaussian_2 + gaussian_3 + velocity_depth
velocity_model_3d = velocity_model_3d / 1000.0

source_location = (random.choice(x), random.choice(y), random.choice(z))

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Travel-time solution
phi = np.ones_like(velocity_model_3d)
source_idx = (int(source_location[0] // delta_x), int(source_location[1] // delta_y), int(source_location[2] // delta_z))
phi[source_idx] = -1.  # Initialize the source location with impossible phi value

T_data = skfmm.travel_time(phi, velocity_model_3d, dx=(delta_x, delta_y, delta_z))

X,Y,Z = torch.meshgrid(torch.linspace(xmin, xmax, nx), torch.linspace(ymin, ymax, ny), torch.linspace(zmin, zmax, nz))
T_data = torch.from_numpy(T_data).float().to(device)


n_receivers = 10
# Generate random x and y coordinates for receivers
x_receivers = np.random.uniform(xmin, xmax, n_receivers)
y_receivers = np.random.uniform(ymin, ymax, n_receivers)
z_receivers = np.ones_like(x_receivers) * zmax  

# Create tensor of receiver positions
receivers_grid_positions = torch.tensor(np.column_stack((x_receivers, y_receivers, z_receivers)), 
                                     dtype=torch.float32, 
                                     device=device)

receivers_idx = []
for i in range(len(x_receivers)):
    idx_x = (abs(x - x_receivers[i])).argmin()
    idx_y = (abs(y - y_receivers[i])).argmin()
    idx_z = (abs(z - z_receivers[i])).argmin()
    receivers_idx.append((idx_x, idx_y, idx_z))

# Get travel times for receiver positions
receivers_travel_times = torch.tensor([[T_data[idx]] for idx in receivers_idx], 
                                    dtype=torch.float32, 
                                    device=device)

# Convert receivers grid positions to tensor
receivers_tensor = receivers_grid_positions.requires_grad_(False)


n_training_points = 1_0_000 
# Define random training points
print(f"X type: {type(X)}")
training_points = np.random.choice(np.arange(Z.numel()), n_training_points, replace=False)
X_trainings = X.reshape(-1, 1)[training_points]
Y_trainings = Y.reshape(-1, 1)[training_points]
Z_trainings = Z.reshape(-1, 1)[training_points]

# Define the Travel Times velocity solutions
velocities = torch.from_numpy(velocity_model_3d).float().to(device)
velocities = velocities.reshape(-1, 1)[training_points].requires_grad_(False).to(device)

# Finally define the torch tensor for the training points
training_points = torch.hstack((X_trainings, Y_trainings, Z_trainings)).requires_grad_(True).to(device)

# grid points for predictions
prediction_grid = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1))).to(device)

params = {'vel': velocities, 'input': training_points, 'data_pts':receivers_tensor, 'data':receivers_travel_times}


def save_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(checkpoint_file, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])

# initialize PINN model and summary
pinn = PINN(L=10, init='kaiming_normal').to(device)
pinn.train()
summary(pinn)
# initialize system
w_eqn = 1.0     # PDE weight
w_heav = 1.0    # Heaviside/constraint weight
w_data = 5.0   # Data/receiver weight (make this bigger if you want data fitting to dominate)

system = LossEikonal(w_eqn=w_eqn, w_data=w_data, w_heav=w_heav)

# summary(pinn)

# maximum iterations
max_iter = 10000
total_loss = np.zeros((max_iter)) 
eqn_loss = np.zeros((max_iter)) 
data_loss = np.zeros((max_iter)) 


# Adam Optimizer
optimizer = torch.optim.Adam(pinn.parameters(), lr=5e-3, betas=(0.5, 0.9))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                       max_iter, 
                                                       eta_min=0, 
                                                       last_epoch=- 1, 
                                                       )

# training start
iterations = tqdm(range(max_iter))
start_time = time.time()
for j in iterations:
    loss, loss_eq, loss_data = system(pinn, params)
    loss.backward()
    optimizer.step()
    scheduler.step()

    optimizer.zero_grad()
    

    total_loss[j] = loss.item()
    eqn_loss[j] = loss_eq.item()
    data_loss[j] = loss_data.item()
    iterations.set_postfix(loss=loss.item())

elapsed = time.time() - start_time                
print('\nTraining time: %.2f' % (elapsed))
# save_checkpoint(pinn, filename="NEW.pth")