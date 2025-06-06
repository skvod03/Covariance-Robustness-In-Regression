# main.py
import numpy as np
from utils.plotting import plot_results
from utils.ModelSimulation import simulate_model

def main():
    """
    The main function to simulate models for Linear Regression, Faulty Linear Regression,
    and Logistic Regression, calculate coverage probabilities for different sample sizes, 
    and plot the results.

    It simulates data using the `simulate_model()` function for each model type (Linear Regression, 
    Faulty Linear Regression, Logistic Regression), for different sample sizes. It calculates the 
    coverage probabilities of three covariance methods (Fisher, Sandwich, Bootstrap) and then 
    plots the results.
    """
    
    # Define parameters
    d, T, B = 10, 200, 200  # Number of features, simulations, and bootstrap resamples
    N = [50, 100, 200, 400]  # Sample sizes to consider
    results = np.zeros((4, 3, 3))  # 4 sample sizes, 3 covariance methods, 3 models

    # Simulate and collect coverage probabilities for Linear Regression, Faulty Linear Regression, and Logistic Regression
    for i in range(4):
        n = N[i]
        
        # Simulate for Linear Regression
        results[i, :, 0] = simulate_model('linreg', d, n, T, B)
        
        # Simulate for Faulty Linear Regression
        results[i, :, 1] = simulate_model('faulty_linreg', d, n, T, B)
        
        # Simulate for Logistic Regression
        results[i, :, 2] = simulate_model('logreg', d, n, T, B)

    # Plot the results for Linear Regression
    plot_results(N, results[:, :, 0], "Coverage Probability vs. Sample Size in Linear Regression")
    
    # Plot the results for Faulty Linear Regression
    plot_results(N, results[:, :, 1], plot_title="Coverage Probability vs. Sample Size in Faulty Linear Regression")
    
    # Plot the results for Logistic Regression
    plot_results(N, results[:, :, 2], plot_title="Coverage Probability vs. Sample Size in Logistic Regression")

# This checks if the script is being run directly and calls the main function
if __name__ == "__main__":
    main()
