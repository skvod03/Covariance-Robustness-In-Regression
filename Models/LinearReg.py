import numpy as np

def LinReg(d, n):
    """
    Simulate a Linear Regression model.
    
    This function generates synthetic data for a linear regression model, where the true underlying
    parameters are sampled from a unit sphere in a d-dimensional space. The data is generated using
    a standard normal distribution, and noise is added to the model. The function then estimates the
    parameters using ordinary least squares (OLS).
    
    Parameters:
    d (int): The number of features (dimensions) in the data.
    n (int): The number of observations (samples).
    
    Returns:
    tuple: 
        - X (numpy.ndarray): A matrix of shape (n, d) representing the generated feature matrix.
        - beta_star (numpy.ndarray): The true parameter vector of shape (d,) sampled from a unit sphere.
        - Y (numpy.ndarray): A vector of observed target values of shape (n,).
        - beta_hat (numpy.ndarray): The estimated parameter vector of shape (d,) using OLS.
    """
    # Generate random feature matrix X with shape (n, d)
    X = np.random.randn(n, d)

    # Sample the true parameters from a unit sphere in R^d
    normal = np.random.randn(d)  
    beta_star = normal / np.linalg.norm(normal)  

    # Generate random noise for the target variable
    epsilon = np.random.randn(n)
    
    # Calculate the target values using the linear model Y = X * beta_star + epsilon
    Y = X @ beta_star + epsilon
    
    # Estimate the parameters using Ordinary Least Squares (OLS)
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    
    return X, beta_star, Y, beta_hat


def FaultyLinReg(d, n):
    """
    Simulate a Faulty Linear Regression model.
    
    This function generates synthetic data where the true model is non-linear, with a quadratic
    term added to the standard linear regression model. The data is generated using a linear combination
    of the features and adding a non-linear component, and noise is added. The function estimates the
    parameters using ordinary least squares (OLS) despite the faulty non-linear model.
    
    Parameters:
    d (int): The number of features (dimensions) in the data.
    n (int): The number of observations (samples).
    
    Returns:
    tuple:
        - X (numpy.ndarray): A matrix of shape (n, d) representing the generated feature matrix.
        - beta_star (numpy.ndarray): The true parameter vector of shape (d,) sampled from a unit sphere.
        - Y (numpy.ndarray): A vector of observed target values of shape (n,).
        - beta_hat (numpy.ndarray): The estimated parameter vector of shape (d,) using OLS (despite non-linearity).
    """
    # Generate random feature matrix X with shape (n, d)
    X = np.random.randn(n, d)

    # Sample the true linear parameters from a unit sphere in R^d
    normal = np.random.randn(d)  
    beta_star = normal / np.linalg.norm(normal)  

    # Sample a second set of parameters for the non-linear term
    normal = np.random.randn(d)  
    theta_star = normal / np.linalg.norm(normal)  

    # Generate random noise for the target variable
    epsilon = np.random.randn(n)
    
    # Non-linear model Y = X * beta_star + (X * theta_star)^2 + epsilon
    Y = X @ beta_star + (X @ theta_star) ** 2 + epsilon
    
    # Estimate the parameters using Ordinary Least Squares (OLS)
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ Y
    
    return X, beta_star, Y, beta_hat
