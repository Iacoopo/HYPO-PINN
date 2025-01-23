import numpy as np
import skfmm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import neuralop
import matplotlib.pyplot as plt
from neuralop.models import FNO3d
import skfmm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ForwardPINO(nn.Module):
    def __init__(self, modes, width):
        super(ForwardPINO, self).__init__()
        self.fourier = FNO3d(
            n_modes_height=modes,
            n_modes_width=modes,
            n_modes_depth=modes,
            hidden_channels=width,
            in_channels=2,  # Velocity model + source grid
            out_channels=1  # Travel times
        )

    def forward(self, velocity_model, source_location):
        # Create a differentiable source grid
        source_grid = torch.zeros_like(velocity_model)
        for i in range(source_location.shape[0]):
            loc = source_location[i].round().long()
            x, y, z = loc[0].item(), loc[1].item(), loc[2].item()
            if (
                0 <= x < source_grid.shape[-3]
                and 0 <= y < source_grid.shape[-2]
                and 0 <= z < source_grid.shape[-1]
            ):
                source_grid[i, 0, x, y, z] = 1  # Mark source location
        x = torch.cat([velocity_model, source_grid], dim=1)
        return self.fourier(x)
    

class InversePINO(nn.Module):
    def __init__(self, modes, width):
        super(InversePINO, self).__init__()
        self.fourier = FNO3d(
            n_modes_height=modes,
            n_modes_width=modes,
            n_modes_depth=modes,
            hidden_channels=width,
            in_channels=2,  # Velocity model + partial travel times
            out_channels=width
        )
        self.fc = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Output is a 3D source location
        )

    def forward(self, velocity_model, partial_travel_time):
        x = torch.cat([velocity_model, partial_travel_time], dim=1)
        x = self.fourier(x)
        x = torch.mean(x, dim=(-3, -2, -1))  # Global pooling
        return self.fc(x)
    

def eikonal_loss_pde(velocity_model, predicted_source_location, forward_pino):
    """
    Compute the PDE-based Eikonal loss for the PINO model.

    Args:
        velocity_model (torch.Tensor): The input velocity model (requires gradients).
        predicted_source_location (torch.Tensor): The predicted source locations.
        forward_pino (nn.Module): The forward PINO model to compute travel times.

    Returns:
        torch.Tensor: The Eikonal loss value.
    """
    # Ensure velocity_model requires gradients
    velocity_model = velocity_model.requires_grad_(True)

    # Compute predicted travel times using the forward PINO model
    predicted_travel_times = forward_pino(velocity_model, predicted_source_location)

    # Compute gradients of travel times with respect to the velocity model
    gradients = torch.autograd.grad(
        outputs=predicted_travel_times,
        inputs=velocity_model,
        grad_outputs=torch.ones_like(predicted_travel_times),  # Backpropagation signal
        create_graph=True,  # Retain computational graph for higher-order gradients
        retain_graph=True   # Retain graph for reuse
    )[0]

    # Compute the squared norm of the gradients
    gradient_norm_squared = torch.sum(gradients ** 2, dim=1)

    # Eikonal loss: (||grad|| - 1 / velocity^2)^2
    velocity_term = 1 / (velocity_model ** 2)
    loss = torch.mean((gradient_norm_squared - velocity_term) ** 2)

    return loss

# ---------------------------- Prepare Data ----------------------------

# Prepare data
train_data, val_data = prepare_data(num_samples, grid_size, noise_std, mask_prob)

# Create Datasets
forward_train_dataset = ForwardPINOData(
    train_data["velocity_models"], train_data["source_locations"], train_data["travel_times"]
)
inverse_train_dataset = InversePINOData(
    train_data["velocity_models"], train_data["partial_travel_times"], train_data["source_locations"]
)

forward_val_dataset = ForwardPINOData(
    val_data["velocity_models"], val_data["source_locations"], val_data["travel_times"]
)
inverse_val_dataset = InversePINOData(
    val_data["velocity_models"], val_data["partial_travel_times"], val_data["source_locations"]
)

# Create DataLoaders
batch_size = 32
forward_train_loader = DataLoader(forward_train_dataset, batch_size=batch_size, shuffle=True)
inverse_train_loader = DataLoader(inverse_train_dataset, batch_size=batch_size, shuffle=True)
forward_val_loader = DataLoader(forward_val_dataset, batch_size=batch_size, shuffle=False)
inverse_val_loader = DataLoader(inverse_val_dataset, batch_size=batch_size, shuffle=False)

print(f"Forward Train Samples: {len(forward_train_dataset)}, Inverse Train Samples: {len(inverse_train_dataset)}")


# ---------------------------- Train FNO "Forward" ----------------------------

def train_forward_pino(
    model, train_loader, val_loader, optimizer, epochs, device, patience=5, factor=0.5
):
    """
    Train the Forward PINO model with ReduceLROnPlateau learning rate scheduler.
    
    Parameters:
        model (nn.Module): The Forward PINO model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        device (torch.device): Device to train on (e.g., "cuda" or "cpu").
        patience (int): Number of epochs with no improvement before reducing LR.
        factor (float): Factor by which the LR is reduced.
    """
    model.to(device)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, verbose=True
    )

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Training loop
        for velocity_model, source_location, true_travel_time in train_loader:
            velocity_model = velocity_model.to(device)
            source_location = source_location.to(device)
            true_travel_time = true_travel_time.to(device)

            optimizer.zero_grad()
            predicted_travel_time = model(velocity_model, source_location)
            loss = criterion(predicted_travel_time, true_travel_time)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}")

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for velocity_model, source_location, true_travel_time in train_loader:
                velocity_model = velocity_model.to(device)
                source_location = source_location.to(device)
                true_travel_time = true_travel_time.to(device)

                predicted_travel_time = model(velocity_model, source_location)
                loss = criterion(predicted_travel_time, true_travel_time)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.6f}")

        # Update the learning rate scheduler
        scheduler.step(val_loss)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}")

        # Stop training if the learning rate becomes too small
        if current_lr < 1e-8:
            print("Learning rate is too small. Stopping training.")
            break



# Parameters
modes, width = 10, 80  # Adjust based on your grid size and model capacity
forward_pino = ForwardPINO(modes, width)

# Optimizer
optimizer = torch.optim.Adam(forward_pino.parameters(), lr=0.001)

# Training
epochs = 50
#train_forward_pino(forward_pino, forward_train_loader, optimizer, epochs, device)
train_forward_pino_with_scheduler(
    forward_pino, forward_train_loader, forward_val_loader, optimizer, epochs=100, device=device
)

#-----------------------------Evaluate Forward PINO-----------------------------

def evaluate_forward_pino(model, dataloader, device):
    """
    Evaluate the Forward PINO model on the validation set.
    
    Parameters:
        model (nn.Module): The Forward PINO model.
        dataloader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to evaluate on (e.g., "cuda" or "cpu").
    
    Returns:
        float: Average loss on the validation set.
    """
    model.to(device)
    model.eval()
    criterion = torch.nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for velocity_model, source_location, true_travel_time in dataloader:
            # Move data to device
            velocity_model = velocity_model.to(device)
            source_location = source_location.to(device)
            true_travel_time = true_travel_time.to(device)

            # Forward pass
            predicted_travel_time = model(velocity_model, source_location)

            # Compute loss
            loss = criterion(predicted_travel_time, true_travel_time)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.6f}")

evaluate_forward_pino(forward_pino, forward_train_loader, device)


# ---------------------------- Train FNO "Inverse" ----------------------------

def train_inverse_pino_with_forward_pino(
    inverse_pino, forward_pino, dataloader, val_loader, epochs, device, initial_lambda_eikonal=0.000001, factor=0.5, patience=5
):
    inverse_pino.to(device)
    forward_pino.to(device)
    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(inverse_pino.parameters(), lr=0.001)
    #optimizer_mse = torch.optim.Adam(inverse_pino.parameters(), lr=0.001)
    #optimizer_eik = torch.optim.Adam(inverse_pino.parameters(), lr=0.00001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, verbose=True
    )

    lambda_eikonal = initial_lambda_eikonal
    
    for epoch in range(epochs):
        inverse_pino.train()
        forward_pino.eval()  # Freeze forward PINO during inverse PINO training
        total_epoch_loss=0
        #epoch_mse = 0
        #epoch_eik = 0

        for velocity_model, partial_true_time, true_source_location in dataloader:
            optimizer.zero_grad()
            #optimizer_mse.zero_grad()
            #optimizer_eik.zero_grad()
            
            velocity_model = velocity_model.to(device)
            partial_true_time = partial_true_time.to(device)
            true_source_location = true_source_location.to(device)


            # Predict source location
            predicted_source_location = inverse_pino(velocity_model, partial_true_time)

            
            # Compute MSE loss for source location
            mse = mse_loss(predicted_source_location, true_source_location)

            # Compute Eikonal loss
            eikonal = eikonal_loss_pde(velocity_model, predicted_source_location, forward_pino)
            #adj_eikonal = lambda_eikonal * eikonal

            # Adjust lambda_eikonal dynamically
            if mse.item() > 0:
                lambda_eikonal = min(lambda_eikonal, mse.item() / (10 * eikonal.item() + 1e-8))

            adj_eikonal = lambda_eikonal * eikonal

            total_loss = mse + adj_eikonal
            total_loss.backward()
            #mse.backward()
            #adj_eikonal.backward()
            
            optimizer.step()
            #if epoch > 20:
            #    optimizer_mse.step()
            #optimizer_eik.step()
            
            total_epoch_loss += total_loss.item()
            #epoch_mse += mse.item()
            #epoch_eik += adj_eikonal.item()
            torch.cuda.empty_cache()

        # Validation loop
        inverse_pino.eval()
        val_loss = 0
        with torch.no_grad():
            for velocity_model, partial_travel_time, true_source_location in val_loader:
                velocity_model = velocity_model.to(device)
                true_source_location = true_source_location.to(device)
                partial_travel_time = partial_travel_time.to(device)

                predicted_source_location = inverse_pino(velocity_model, partial_travel_time)
                ms_eval = mse_loss(predicted_source_location, true_source_location)
                #eik_eval = eikonal_loss_pde(velocity_model, predicted_source_location, forward_pino)
                #adj_eikonal = lambda_eikonal * eik_eval
                #total_val_loss = adj_eikonal + ms_eval
                val_loss += ms_eval.item()
                torch.cuda.empty_cache()
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.6f}")

        # Update the learning rate scheduler
        scheduler.step(val_loss)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}")

        # Stop training if the learning rate becomes too small
        if current_lr < 1e-8:
            print("Learning rate is too small. Stopping training.")
            break


        print(f"Epoch {epoch + 1}/{epochs}, MSE Loss: {mse / len(dataloader):.6f}, "
              f"Eikonal Loss: {adj_eikonal / len(dataloader):.6f}")
        

modes, width = 20, 180  
inverse_pino = InversePINO(modes, width)


epochs = 50
train_inverse_pino_with_forward_pino(
    inverse_pino=inverse_pino, forward_pino=forward_pino, dataloader=inverse_train_loader, val_loader=inverse_val_loader, epochs=50, device=device
)        

# ---------------------------- Evaluate Inverse PINO ----------------------------


def display_source_locations(true_locations, predicted_locations):
    """
    Display true source locations vs predicted source locations.
    
    Parameters:
        true_locations (torch.Tensor): True source locations of shape [N, 3].
        predicted_locations (torch.Tensor): Predicted source locations of shape [N, 3].
    """
    true_locations = true_locations.cpu().numpy()
    predicted_locations = predicted_locations.cpu().detach().numpy()
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter true locations
    ax.scatter(true_locations[:, 0], true_locations[:, 1], true_locations[:, 2], 
               c='blue', label='True Source Locations', marker='o')

    # Scatter predicted locations
    ax.scatter(predicted_locations[:, 0], predicted_locations[:, 1], predicted_locations[:, 2], 
               c='red', label='Predicted Source Locations', marker='^')

    # Connect true and predicted locations with lines
    for true, pred in zip(true_locations, predicted_locations):
        ax.plot([true[0], pred[0]], [true[1], pred[1]], [true[2], pred[2]], 
                c='gray', linestyle='dashed', linewidth=0.5)

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('True vs Predicted Source Locations')
    ax.legend()
    plt.show()

# Example batch from validation dataset
batch_size = 32  # Adjust to desired batch size for visualization
val_loader = DataLoader(inverse_val_dataset, batch_size=batch_size, shuffle=False)

# Fetch a batch of data
velocity_model, partial_travel_time, true_source_location = next(iter(val_loader))

# Move data to the appropriate device
velocity_model = velocity_model.to(device)
partial_travel_time = partial_travel_time.to(device)
true_source_location = true_source_location.to(device)

# Model evaluation
inverse_pino.eval()
with torch.no_grad():
    predicted_source_location = inverse_pino(velocity_model, partial_travel_time)

# Display true vs predicted source locations
display_source_locations(true_source_location, predicted_source_location)