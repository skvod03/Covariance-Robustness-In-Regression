import matplotlib.pyplot as plt
import os

def plot_results(N, results, plot_title):
    """
    Plot the results of the simulations with respect to sample size and coverage probabilities.

    Args:
        N (list): List of sample sizes.
        results (numpy array): Coverage probabilities for each method (fisher, sandwich, bootstrap).
        plot_title (str): The title of the plot.
        model_name (str): The model name used in the simulation (e.g., 'Linear Regression', 'Logistic Regression').
    """
    plt.figure(figsize=(8, 5))
    plt.plot(N, results[:, 0], marker='o', linestyle='-', label='Fisher Covariance')
    plt.plot(N, results[:, 1], marker='s', linestyle='-', label='Sandwich Covariance')
    plt.plot(N, results[:, 2], marker='^', linestyle='-', label='Bootstrap Covariance')

    plt.xlabel("Sample Size (n)")
    plt.ylabel("Coverage Probability")
    plt.title(f"{plot_title}")
    plt.axhline(y=0.9, color='gray', linestyle='dashed', label='90% Confidence Level')
    plt.legend()
    plt.grid()

    # Ensure the 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Save the plot to the 'plots' folder
    plot_filename = f"{plot_title.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(plot_filename)

    # Optionally, display the plot (can be removed if not needed)
    plt.show()
