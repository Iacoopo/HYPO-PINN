import numpy as np
import skfmm
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import neuralop
from neuralop.models import FNO3d
import skfmm
import numpy as np

#---------------Fourier Neural Operator----------------

class FNOInverse(nn.Module):
    def __init__(self, modes, width):
        super(FNOInverse, self).__init__()
        self.fourier = FNO3d(
            n_modes_height=modes,
            n_modes_width=modes,
            n_modes_depth=modes,
            hidden_channels=width,
            in_channels=2,  # Velocity model + partial travel time
            out_channels=width
        )
        self.fc = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # Output is a 3D source location
        )

    def forward(self, velocity_model, partial_true_time):
        # Concatenate inputs as channels
        x = torch.cat([velocity_model, partial_true_time], dim=1)
        
        # Fourier Neural Operator for feature extraction
        x = self.fourier(x)
        
        # Global pooling and fully connected layers
        x = torch.mean(x, dim=(-3, -2, -1))  # Global pooling over spatial dimensions
        x = self.fc(x)
        return x
    
#---------------Data Preparation----------------

velocity_models, partial_travel_times, source_locations = prepare_data(
    num_samples, grid_size, noise_std, mask_prob
)

# Split into Train and Validation Sets
split_ratio = 0.8
num_train = int(split_ratio * num_samples)

train_velocity_models = velocity_models[:num_train]
train_partial_times = partial_travel_times[:num_train]
train_source_locations = source_locations[:num_train]

val_velocity_models = velocity_models[num_train:]
val_partial_times = partial_travel_times[num_train:]
val_source_locations = source_locations[num_train:]

# Create Datasets and DataLoaders
train_dataset = TravelTimeDataset(train_velocity_models, train_partial_times, train_source_locations)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TravelTimeDataset(val_velocity_models, val_partial_times, val_source_locations)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


#---------------Training----------------

def train_inverse_fno(model, dataloader, optimizer, epochs, device, patience=10, factor=0.6):
    model.to(device)
    criterion = nn.MSELoss()    
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6, last_epoch=-1, verbose='deprecated')
    # Early stopping parameters
    early_stopping_patience = 100
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_path = "/kaggle/working/best_model.pth"
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for velocity_model, partial_true_time, true_source_location in dataloader:
            velocity_model, partial_true_time, true_source_location = (
                velocity_model.to(device),
                partial_true_time.to(device),
                true_source_location.to(device),
            )
    
            # Forward pass
            predicted_source_location = model(velocity_model, partial_true_time)
    
            # Compute loss
            loss = criterion(predicted_source_location, true_source_location)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            epoch_loss += loss.item()
    
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.6f}")
    
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for velocity_model, source_location, true_travel_time in val_loader:
                velocity_model = velocity_model.to(device)
                source_location = source_location.to(device)
                true_travel_time = true_travel_time.to(device)
    
                predicted_travel_time = model(velocity_model, source_location)
                loss = criterion(predicted_travel_time, true_travel_time)
                val_loss += loss.item()
    
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.6f}")
    
        # Update the learning rate scheduler
        #scheduler.step(val_loss)
        scheduler.step()
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6e}")
    
        # Check early stopping condition
        if val_loss < best_val_loss and epoch > 40:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print("Validation loss improved. Saving the best model.")
            # Save the best model
            torch.save(model, best_model_path)
        else:
            if epoch >50:
                epochs_without_improvement += 1
                print(f"No improvement in validation loss for {epochs_without_improvement} epochs.")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered. No improvement in validation loss for {early_stopping_patience} consecutive epochs.")
            break
    
        # Stop training if the learning rate becomes too small
        if current_lr < 1e-8:
            print("Learning rate is too small. Stopping training.")
            break




# Model initialization
modes = grid_size
width = modes * 8 
inverse_fno = FNOInverse(modes, width)
optimizer = torch.optim.Adam(inverse_fno.parameters(), lr=0.01)

# Training
train_inverse_fno(inverse_fno, train_loader, optimizer, epochs=200, device=device)

#---------------Evaluation----------------

def evaluate_inverse_fno(model, dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for velocity_model, partial_true_time, true_source_location in dataloader:
            velocity_model, partial_true_time, true_source_location = (
                velocity_model.to(device),
                partial_true_time.to(device),
                true_source_location.to(device),
            )

            predicted_source_location = model(velocity_model, partial_true_time)
            loss = criterion(predicted_source_location, true_source_location)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.6f}")
    
inverse_fno = torch.load("/working/best_model.pth")
evaluate_inverse_fno(inverse_fno, val_loader, device)

#---------------Visualization----------------

import matplotlib.pyplot as plt

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
    plt.savefig("/kaggle/working/HD.png", dpi=900, bbox_inches='tight', transparent=False)


    plt.show()

# Example batch from validation dataset
velocity_model, partial_travel_time, true_source_location = next(iter(val_loader))

# Move data to the appropriate device
velocity_model = velocity_model.to(device)
partial_travel_time = partial_travel_time.to(device)
true_source_location = true_source_location.to(device)

# Model evaluation
inverse_fno.eval()
with torch.no_grad():
    predicted_source_location = inverse_fno(velocity_model, partial_travel_time)

# Display true vs predicted source locations
display_source_locations(true_source_location, predicted_source_location)