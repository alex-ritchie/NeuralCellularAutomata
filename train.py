# amr 2024
# Display Utils for Neural Cellular Automata
# adapted from: 
# blog - https://distill.pub/2020/growing-ca/
# code - https://colab.research.google.com/github/google-research/self-organising-systems/blob/master/notebooks/growing_ca.ipynb#scrollTo=zR6I1JONmWBb
# amr 2024
# Training script for Neural Cellular Automata using PyTorch

import torch
import torch.optim as optim
import numpy as np
import wandb  # Import WandB
from neuralca import NeuralCA, make_seed, visualize_batch, plot_loss

# Initialize WandB
wandb.init(project="neural-cellular-automata")

# Hyperparameters
#@title Cellular Automata Parameters
CHANNEL_N = 16        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 0.5

# Target Pattern
TARGET_EMOJI = "🦎" #@param {type:"string"}

# Experiment Parameters
EXPERIMENT_TYPE = "Regenerating" #@param ["Growing", "Persistent", "Regenerating"]
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = [0, 1, 1][EXPERIMENT_N]
DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

# Training Parameters
NUM_EPOCHS = 1000  # Number of training epochs
LEARNING_RATE = 0.001  # Learning rate
SEED_SIZE = 64  # Size of the seed
BATCH_SIZE = 32  # Batch size

# Log hyperparameters
wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "channel_n": CHANNEL_N,
    "cell_fire_rate": CELL_FIRE_RATE
}

# Initialize the model
model = neuralCA(channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Create a seed for the cellular automata
seed = make_seed(SEED_SIZE, BATCH_SIZE)
seed_tensor = torch.tensor(seed, dtype=torch.float32)

# Training loop
loss_log = []
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()  # Zero the gradients

    # Forward pass
    output = model(seed_tensor)

    # Calculate loss (example: mean squared error)
    loss = F.mse_loss(output, seed_tensor)  # Adjust target as needed
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    loss_log.append(loss.item())

    # Log loss to WandB
    wandb.log({"epoch": epoch, "loss": loss.item()})

    # Log gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"gradients/{name}": param.grad.detach().cpu().numpy()})

    # Log model parameters
    for name, param in model.named_parameters():
        wandb.log({f"parameters/{name}": param.detach().cpu().numpy()})

    # Log additional metrics (e.g., mean and std of gradients)
    if len(loss_log) > 0:
        wandb.log({
            "mean_loss": np.mean(loss_log),
            "std_loss": np.std(loss_log),
        })

    # Print progress
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{NUM_EPOCHS}], Loss: {loss.item():.4f}')

    # Visualize batch every 100 epochs
    if epoch % 100 == 0:
        visualize_batch(seed_tensor, output, epoch)

# Plot loss history
plot_loss(loss_log)

# Finish WandB run
wandb.finish()