
# Hypocenter Prediction Using PINN, FNO, and PINO

This repository contains implementations of models for predicting the hypocenter of seismic events based on recorded travel times at receivers. These models utilize advanced machine learning techniques, including Physics-Informed Neural Networks (PINN), Fourier Neural Operators (FNO), and Physics-Informed Neural Operators (PINO).

## Repository Overview

The repository includes the following files:

* **`PINN.py`** : Implementation of a Physics-Informed Neural Network (PINN) that enforces the Eikonal equation as a physical constraint while learning to predict travel times and hypocenters.
* **`FNO.py`** : Implementation of a Fourier Neural Operator (FNO) for hypocenter prediction using global frequency domain representation.
* **`PINO.py`** : Implementation of a Physics-Informed Neural Operator (PINO) that integrates domain knowledge with neural operator frameworks for enhanced prediction accuracy.

## Features

* **Physics-Informed Learning** : The models incorporate domain knowledge through physical laws (e.g., the Eikonal equation), enabling accurate predictions with limited data.
* **Neural Operator Frameworks** : FNO and PINO enable efficient learning of high-dimensional mappings, making them suitable for complex seismic velocity models.
* **3D Hypocenter Localization** : Predicts the three-dimensional source location of seismic events.

## Installation

To use the models, ensure the following dependencies are installed:

* Python 3.8+
* PyTorch
* NumPy
* SciPy
* Matplotlib
* Neural Operator library (`neuralop`)
* scikit-fmm (`skfmm`)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

Each script includes routines to prepare synthetic data, including:

* **Velocity Models** : 3D seismic velocity grids.
* **Travel Times** : Synthetic travel time fields computed using Fast Marching Methods.
* **Receiver Data** : Partial travel times recorded at predefined receiver locations.

### Training Models

Run the respective scripts to train models:

* Train PINN:
  ```bash
  python PINN.py
  ```
* Train FNO:
  ```bash
  python FNO.py
  ```
* Train PINO:
  ```bash
  python PINO.py
  ```

### Visualization

The scripts include visualization utilities to compare true and predicted source locations in 3D.

### Evaluation

Evaluate models on validation datasets by running the corresponding evaluation functions included in each script.

## Models Summary

1. **PINN** : Implements physics-informed constraints directly in the loss function for accurate learning with limited data.
2. **FNO** : Leverages Fourier transformations to efficiently learn mappings from velocity models to hypocenters.
3. **PINO** : Extends FNO with additional physical constraints for robust predictions in noisy environments.

## Examples

* Generate synthetic seismic data with varying velocity models and noise levels.
* Predict hypocenters and compare model performance using validation metrics like MSE.

## Citation

If you use this repository in your research, please cite the relevant papers and frameworks that inspired its development.
