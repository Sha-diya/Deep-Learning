import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Generate synthetic dataset
X, y = make_classification(
    n_samples=100, 
    n_features=2, 
    n_informative=1, 
    n_redundant=0, 
    n_classes=2, 
    n_clusters_per_class=1,
    random_state=41, 
    hypercube=False, 
    class_sep=10
)


#plt.figure(figsize=(10,6))
#plt.scatter(X[:,0],X[:,1], c=y,s=100)

def step(z):
    return 1 if z>0 else 0

def perceptron(X, y):
    m = []  # List to store slope (m) values over iterations
    b = []  # List to store intercept (b) values over iterations
    X = np.insert(X, 0, 1, axis=1)  # Add bias term to feature matrix

    weights = np.ones(X.shape[1])  # Initialize weights to 1
    lr = 0.1  # Learning rate

    for i in range(1000):  # Iterate 1000 times
        j = np.random.randint(0, 100)  # Randomly select a sample
        y_hat = step(np.dot(X[j], weights))  # Predict using current weights
        weights = weights + lr * (y[j] - y_hat) * X[j]  # Update weights

        # Calculate slope (m) and intercept (b) from updated weights
        m.append(-(weights[1] / weights[2]))
        b.append(-(weights[0] / weights[2]))

    return m, b  # Return all slopes and intercepts over time

m,b=perceptron(X, y)

fig, ax = plt.subplots(figsize=(9, 5))  # Create a plot figure and axis
x_i = np.arange(-3, 3, 0.1)  # Generate x-axis values from -3 to 3
y_i = x_i * m[0] + b[0]  # Calculate initial decision boundary (y = mx + b)
ax.scatter(X[:, 0], X[:, 1], c=y, s=100)  # Plot the data points
line, = ax.plot(x_i, y_i, 'r-', linewidth=2)  # Plot initial decision boundary
plt.ylim(-3, 3)  # Set y-axis limits

def update(i):
    label = 'epoch {0}'.format(i+1)# Create a label for the current epoch
    line.set_ydata(x_i*m[i] + b[i])# Update the decision boundary
    ax.set_xlabel(label)# Update the x-axis label
    
anim = FuncAnimation(fig, update, repeat=True,frames=200,interval=100)

