# HYPO-PINN

## Overview

This project implements a Physics-Informed Neural Network (PINN) for hypocenter estimation. The goal is to combine machine learning techniques with domain-specific physical laws to create a robust system for estimating earthquake hypocenters using seismic data.

The notebook includes Python code, explanations, and results showcasing the development and application of the PINN model. This project leverages PyTorch for the neural network and introduces domain-specific formulations to ensure the model adheres to physical principles.

---

## Features

* **Physics-Informed Neural Network** : Integrates physical constraints into the loss function.
* **Hypocenter Estimation** : Focused on solving geophysical problems related to earthquake hypocenter localization.
* **PyTorch-Based Implementation** : Uses PyTorch for ease of customization and scalability.
* **Visualizations** : Includes explanations and visualizations to aid understanding.

---

## PINN Structure and Technical Details

The Physics-Informed Neural Network (PINN) in this project is designed as follows:

### Network Architecture

* The network comprises several fully connected layers with non-linear activation functions.
* Input features include seismic data attributes such as arrival times and station coordinates.
* The output predicts the hypocenter coordinates (latitude, longitude, depth) and the origin time.

### Loss Functions

* **Physics-Based Loss** : Ensures the network predictions adhere to the seismic wave propagation equations derived from the physics of wave travel.
* **Data Loss** : Compares predicted values with observed data (e.g., arrival times) using mean squared error.
* **Regularization Loss** : Helps prevent overfitting by penalizing large weights in the network.

### Training Details

* The model is trained using backpropagation with gradient descent optimizers (e.g., Adam).
* A combination of synthetic and observed seismic data is used for training and validation.
* Batch normalization and dropout layers are incorporated to improve generalization.

---

## Results

The project demonstrates:

* Accurate hypocenter localization using the PINN approach.
* Comparisons with traditional methods.
* Visualization of model predictions and error distributions.

---

## Acknowledgments

* Theoretical basis inspired by [Laplace Hypopinn](https://arxiv.org/abs/2205.14439).

---

Feel free to reach out if you have any questions or issues!
