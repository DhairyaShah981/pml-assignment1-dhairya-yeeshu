import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate synthetic data using PyTorch
data_size = 100
true_loc = 2.0
true_scale = 4.0
data = Normal(true_loc, true_scale).sample((data_size,))

# Define parameter ranges
loc_range = torch.linspace(-10, 14, 300)
scale_range = torch.linspace(0.1, 20, 300)

# Calculate log-likelihood values for each parameter combination
log_likelihood_matrix = torch.zeros((len(loc_range), len(scale_range)))
for i, loc in enumerate(loc_range):
    for j, scale in enumerate(scale_range):
        log_likelihood_matrix[i, j] = Normal(loc, scale).log_prob(data).sum()

# Plot the contour plot
plt.figure(figsize=(10, 6))
contour = plt.contourf(scale_range, loc_range, log_likelihood_matrix.T, cmap='viridis')
plt.colorbar(contour, label='Log-Likelihood')
plt.xlabel('Scale')
plt.ylabel('Loc')
plt.title('2D Contour Plot of Log-Likelihood')
plt.show()

# Parameters to be estimated
loc = torch.tensor(0.0, requires_grad=True)
scale = torch.tensor(1.0, requires_grad=True)

# Learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Optimization loop
loss_history = []

for i in range(num_iterations):
    # Compute the negative log-likelihood loss
    loss = -Normal(loc, scale).log_prob(data).sum()
    
    # Backpropagation and gradient descent
    loss.backward()
    with torch.no_grad():
        loc -= learning_rate * loc.grad
        scale -= learning_rate * scale.grad
        loc.grad.zero_()
        scale.grad.zero_()
    
    loss_history.append(loss.item())

# Plot the convergence of loss
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood')
plt.title('Convergence of MLE using Gradient Descent')
plt.show()

# Print the estimated MLE parameters
print(f"Estimated loc: {loc.item():.4f}")
print(f"Estimated scale: {scale.item():.4f}")


# Parameters to be estimated
loc = torch.tensor(0.0, requires_grad=True)
log_scale = torch.tensor(0.0, requires_grad=True)  # Learn log(scale)

# Learning rate and number of iterations
learning_rate = 0.01
num_iterations = 1000

# Optimization loop
loss_history = []

for i in range(num_iterations):
    # Compute the negative log-likelihood loss
    loss = -Normal(loc, torch.exp(log_scale)).log_prob(data).sum()
    
    # Backpropagation and gradient descent
    loss.backward()
    with torch.no_grad():
        loc -= learning_rate * loc.grad
        log_scale -= learning_rate * log_scale.grad
        loc.grad.zero_()
        log_scale.grad.zero_()
    
    loss_history.append(loss.item())

# Transform log_scale to get the estimated scale
estimated_scale = torch.exp(log_scale)

# Plot the convergence of loss
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood')
plt.title('Convergence of MLE using Gradient Descent')
plt.show()

# Print the estimated MLE parameters
print(f"Estimated loc: {loc.item():.4f}")
print(f"Estimated scale: {estimated_scale.item():.4f}")
