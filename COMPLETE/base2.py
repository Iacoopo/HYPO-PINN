import numpy as np
import skfmm
import torch
import torch.nn as nn
import torch.fft
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle


# Fourier Neural Operator (FNO)
class FNO3D(nn.Module):
    def __init__(self, modes, width):
        super(FNO3D, self).__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Conv3d(1, self.width, kernel_size=1)
        self.fourier_layers = nn.ModuleList([FourierLayer3D(self.modes, self.width) for _ in range(4)])
        self.fc1 = nn.Conv3d(self.width, 128, kernel_size=1)
        self.fc2 = nn.Conv3d(128, 1, kernel_size=1)
        self.upsample = nn.Upsample(size=(10, 10, 10), mode='trilinear', align_corners=False)

    def forward(self, x):
        x = self.fc0(x)
        for layer in self.fourier_layers:
            x = layer(x)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        x = self.upsample(x)
        return x


class FourierLayer3D(nn.Module):
    def __init__(self, modes, width):
        super(FourierLayer3D, self).__init__()
        self.modes = modes
        self.width = width
        self.weights = nn.Parameter(
            torch.randn(width, width, self.modes, self.modes, self.modes, dtype=torch.cfloat) * 0.01
        )

    def forward(self, x):
        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))
        x_ft = x_ft[..., :self.modes, :self.modes, :self.modes]
        x_ft = torch.einsum("bcxyz,coxyz->boxyz", x_ft, self.weights)
        x = torch.fft.ifftn(x_ft, dim=(-3, -2, -1)).real
        return x


# Travel Time Dataset
class TravelTimeDataset(Dataset):
    def __init__(self, velocity_model, grid_size, num_samples, save_path=None):
        self.velocity_model = velocity_model.astype(np.float32)
        self.grid_size = grid_size
        self.num_samples = num_samples
        self.samples = []

        if save_path and self.load_dataset(save_path):
            print("Loaded dataset from disk.")
        else:
            self.generate_samples()
            if save_path:
                self.save_dataset(save_path)

    def generate_samples(self):
        print("Generating dataset...")
        for _ in tqdm(range(self.num_samples), desc="Generating samples"):
            source = np.random.randint(0, self.grid_size, size=3)
            travel_times = self.simulate_travel_times(source)  # Full grid
            receivers = np.random.randint(0, self.grid_size, size=(10, 3))
            sparse_grid = self.simulate_sparse_grid(source, receivers, travel_times)
            self.samples.append((sparse_grid, travel_times))  # Store sparse grid and full grid

    def simulate_travel_times(self, source):
        grid = np.ones_like(self.velocity_model, dtype=np.float32)
        grid[tuple(source)] = -1
        travel_time_field = skfmm.travel_time(grid, 1 / self.velocity_model)
        return travel_time_field

    def simulate_sparse_grid(self, source, receivers, travel_times):
        grid = np.full_like(self.velocity_model, fill_value=-1e4, dtype=np.float32)
        grid[tuple(source)] = 0.0
        for receiver in receivers:
            grid[tuple(receiver)] = travel_times[tuple(receiver)]
        return grid

    def save_dataset(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"Dataset saved to {save_path}")

    def load_dataset(self, save_path):
        try:
            with open(save_path, 'rb') as f:
                self.samples = pickle.load(f)
            return True
        except FileNotFoundError:
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sparse_grid, full_grid = self.samples[idx]
        sparse_grid = torch.tensor(sparse_grid, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        full_grid = torch.tensor(full_grid, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        return sparse_grid, full_grid


# Training Loop
def train_model(fno, dataloader, optimizer, loss_fn, epochs, device):
    for epoch in range(epochs):
        fno.train()
        epoch_loss = 0
        for sparse_grid, full_grid in dataloader:
            optimizer.zero_grad()
            inputs = sparse_grid.to(device)
            targets = full_grid.to(device)

            # Predict the full travel time field
            predictions = fno(inputs)

            # Compute MSE loss between predicted and full target grid
            loss = loss_fn(predictions, targets)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")


# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    velocity_model = np.ones((10, 10, 10)) * 5.0
    num_samples = 1000
    grid_size = 10
    save_path = "travel_time_dataset.pkl"

    # Initialize dataset and dataloader
    dataset = TravelTimeDataset(velocity_model, grid_size, num_samples, save_path=save_path)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize FNO and optimizer
    fno = FNO3D(modes=8, width=32).to(device)
    optimizer = optim.Adam(fno.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Train the model
    train_model(fno, dataloader, optimizer, loss_fn, epochs=10, device=device)
