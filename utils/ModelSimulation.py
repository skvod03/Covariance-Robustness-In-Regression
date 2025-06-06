from Covarience.LinearModelCov import linear_fisher_cov, linear_sandwich_cov, linear_bootstrap_cov
from Covarience.LogisticModelCov import logistic_fisher_cov, logistic_sandwich_cov, logistic_bootstrap_cov
from Models.LinearReg import LinReg, FaultyLinReg
from Models.LogisticReg import LogReg
import scipy.stats as stats
import numpy as np

def in_treshhold(beta, beta_hat, Sigma, alpha=0.1):
    """
    Check if the estimated coefficient vector (beta_hat) is within the confidence region
    defined by the covariance matrix (Sigma) using a chi-square threshold.

    Parameters:
    beta (numpy.ndarray): The true coefficient vector (d-dimensional).
    beta_hat (numpy.ndarray): The estimated coefficient vector (d-dimensional).
    Sigma (numpy.ndarray): The covariance matrix (d x d).
    alpha (float): The significance level, default is 0.1.

    Returns:
    bool: True if the point is inside the confidence region, False otherwise.
    """
    d = len(beta_hat)
    # Compute chi-square threshold for a confidence level of (1 - alpha)
    chi2_threshold = stats.chi2.ppf(1 - alpha, df=d)
    
    # Compute the Mahalanobis distance and check if it's within the chi-square threshold
    diff = beta - beta_hat  # Compute (β - β_hat)
    return diff.T @ np.linalg.inv(Sigma) @ diff <= chi2_threshold


def simulate_model(model_type, d, n, T, B):
    """
    Simulates and collects coverage probabilities for the given model type (Linear, Faulty Linear, or Logistic regression).
    The simulation is repeated T times for each model, and the coverage probability is computed for each of the 
    covariance methods (Fisher, Sandwich, Bootstrap).

    Parameters:
    model_type (str): The type of model to simulate ('linreg', 'faulty_linreg', or 'logreg').
    d (int): The number of features.
    n (int): The number of data points.
    T (int): The number of simulations to run.
    B (int): The number of bootstrap resamples.

    Returns:
    tuple: Coverage probabilities for the Fisher, Sandwich, and Bootstrap covariance methods for the specified model type.
    """
    linear_model = {'linreg': LinReg, 'faulty_linreg': FaultyLinReg}
    
    # Initialize counters for coverage probabilities
    fisher_count = 0
    sand_count = 0
    boot_count = 0

    for _ in range(T):
        # Simulate data and calculate covariance for logistic regression model
        if model_type == 'logreg':
            X, beta_star, Y, beta_hat, p = LogReg(d, n)
            
            # Compute the Fisher covariance and check coverage
            sigma = logistic_fisher_cov(X, p)
            fisher_count += int(in_treshhold(beta_star, beta_hat, sigma, alpha=0.1))

            # Compute the Sandwich covariance and check coverage
            sigma = logistic_sandwich_cov(X, Y, beta_hat)
            sand_count += int(in_treshhold(beta_star, beta_hat, sigma, alpha=0.1))

            # Compute the Bootstrap covariance and check coverage
            sigma = logistic_bootstrap_cov(B, X, Y)
            boot_count += int(in_treshhold(beta_star, beta_hat, sigma, alpha=0.1))

        else:
            # Simulate data for linear or faulty linear regression models
            X, beta_star, Y, beta_hat = linear_model[model_type](d, n)

            # Compute the Fisher covariance and check coverage
            sigma = linear_fisher_cov(X, Y, beta_hat)
            fisher_count += int(in_treshhold(beta_star, beta_hat, sigma, alpha=0.1))

            # Compute the Sandwich covariance and check coverage
            sigma = linear_sandwich_cov(X, Y, beta_hat)
            sand_count += int(in_treshhold(beta_star, beta_hat, sigma, alpha=0.1))

            # Compute the Bootstrap covariance and check coverage
            sigma = linear_bootstrap_cov(B, X, Y, beta_hat)
            boot_count += int(in_treshhold(beta_star, beta_hat, sigma, alpha=0.1))

    # Return the coverage probabilities for Fisher, Sandwich, and Bootstrap methods
    return fisher_count / T, sand_count / T, boot_count / T
