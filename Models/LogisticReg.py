from sklearn.linear_model import LogisticRegression
import numpy as np

def LogReg(d, n):
    """
    Simulate a Logistic Regression model.

    This function generates synthetic data for a logistic regression model. The true underlying parameters 
    are sampled from a unit sphere in a d-dimensional space. The data is generated using a logistic model 
    and a binary response variable. The function then estimates the parameters using logistic regression 
    and computes the predicted probabilities for the logistic model.

    Parameters:
    d (int): The number of features (dimensions) in the data.
    n (int): The number of observations (samples).

    Returns:
    tuple:
        - X (numpy.ndarray): A matrix of shape (n, d) representing the generated feature matrix.
        - beta_star (numpy.ndarray): The true parameter vector of shape (d,) sampled from a unit sphere.
        - Y (numpy.ndarray): A vector of binary observed target values of shape (n,).
        - beta_hat (numpy.ndarray): The estimated parameter vector of shape (d,) from logistic regression.
        - prob_zero_hat (numpy.ndarray): The predicted probabilities for the logistic regression model at each observation.
    """
    # Generate random feature matrix X with shape (n, d)
    X = np.random.randn(n, d)

    # Sample the true parameters from a unit sphere in R^d
    normal = np.random.randn(d)  
    beta_star = normal / np.linalg.norm(normal)  

    # Generate probabilities for the binary logistic regression model
    prob_zero = 1 / (1 + np.exp(X @ beta_star)) 
    
    # Generate binary outcomes based on the logistic probabilities (Bernoulli trials)
    Y = np.random.binomial(n=1, p=1 - prob_zero, size=n)  
    
    # Fit a logistic regression model using the generated data
    log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    log_reg.fit(X, Y)
    
    # Extract the estimated coefficients from the fitted model
    beta_hat = log_reg.coef_[0]
    
    # Calculate predicted probabilities for the logistic regression model
    prob_zero_hat = 1 / (1 + np.exp(X @ beta_hat))
    
    return X, beta_star, Y, beta_hat, prob_zero_hat
