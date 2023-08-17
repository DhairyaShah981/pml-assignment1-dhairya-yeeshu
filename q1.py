import torch
import matplotlib.pyplot as plt
import numpy as np
# Initialize parameters and learning rate
theta = torch.tensor([0.0, 0.0], requires_grad=True)
learning_rate = 0.07

num_epochs = 100
theta0_history = []  # To store the values of theta0 at each iteration
theta1_history = []  # To store the values of theta1 at each iteration
loss_history = [] 

for epoch in range(num_epochs):
    f = (theta[0] - 2)**2 + (theta[1] - 3)**2
    loss_history.append(f.item())
    theta0_history.append(theta[0].item())  # Append current theta0 to history
    theta1_history.append(theta[1].item())  # Append current theta1 to history
    
    f.backward()
    
    with torch.no_grad():
        theta -= learning_rate * theta.grad
    
    theta.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {f.item():.4f}, Theta: {theta.tolist()}')

#Plot the convergence of thetas
# plt.plot(theta0_history, label='Theta 0')
# plt.plot(theta1_history, label='Theta 1')
# plt.xlabel('Iteration')
# plt.ylabel('Parameter Value')
# plt.title('Convergence of Parameters')
# plt.legend()
# plt.show()


# # Plot the convergence
# plt.plot(loss_history, label='Loss')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# plt.title('Convergence of Gradient Descent')
# plt.legend()
# plt.show()


# Create a contour plot
# theta0_range = np.linspace(-5, 5, 400)
# theta1_range = np.linspace(-5, 5, 400)
# theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)
# f_grid = (theta0_grid - 2)**2 + (theta1_grid - 3)**2

# plt.figure(figsize=(8, 6))
# plt.contour(theta0_grid, theta1_grid, f_grid, levels=20, colors='gray')
# plt.scatter(theta0_history, theta1_history, c='r', label='Convergence path')
# plt.plot(theta0_history, theta1_history, c='r')
# plt.plot(theta[0].detach().numpy(), theta[1].detach().numpy(), marker='o', markersize=8, label='Optimized point', c='b')
# plt.xlabel(r'$\theta_0$')
# plt.ylabel(r'$\theta_1$')
# plt.title('Gradient Descent Convergence')
# plt.legend()
# plt.colorbar()
# plt.show()



theta0_range = np.linspace(-5, 5, 400)
theta1_range = np.linspace(-5, 5, 400)
theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)
f_grid = (theta0_grid - 2)**2 + (theta1_grid - 3)**2

plt.figure(figsize=(8, 6))
contour = plt.contourf(theta0_grid, theta1_grid, f_grid, levels=20, cmap='viridis')
plt.colorbar(contour, label='Function Value')
plt.scatter(theta0_history, theta1_history, c='r', label='Convergence path')
plt.plot(theta0_history, theta1_history, c='r')
plt.plot(theta[0].detach().numpy(), theta[1].detach().numpy(), marker='o', markersize=8, label='Optimized point', c='b')
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.show()




