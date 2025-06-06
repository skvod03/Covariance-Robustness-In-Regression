import numpy as np
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed

def logistic_fisher_cov(X, p):
    """
    Compute the Fisher covariance matrix for logistic regression.

    The Fisher covariance matrix is computed using the inverse of the weighted Gram matrix 
    (X^T W X), where W is a diagonal matrix of weights based on the predicted probabilities.

    Parameters:
    X (numpy.ndarray): The design matrix of shape (n, d), where n is the number of observations 
                        and d is the number of features.
    p (numpy.ndarray): The predicted probabilities of shape (n,), representing the probability 
                        of Y=1.

    Returns:
    numpy.ndarray: The Fisher covariance matrix of shape (d, d).
    """
    return np.linalg.inv(X.T @ np.diag(p * (1 - p)) @ X)  # Fisher covariance matrix


def logistic_sandwich_cov(X, Y, beta_hat):
    """
    Compute the Sandwich covariance matrix for logistic regression.

    The Sandwich estimator accounts for the heteroscedasticity of residuals and is calculated 
    using the Hessian matrix and the gradients of the log-likelihood function.

    Parameters:
    X (numpy.ndarray): The design matrix of shape (n, d), where n is the number of observations 
                        and d is the number of features.
    Y (numpy.ndarray): The target vector of shape (n,), representing the binary observed values (0 or 1).
    beta_hat (numpy.ndarray): The estimated coefficients of the model of shape (d,).

    Returns:
    numpy.ndarray: The Sandwich covariance matrix of shape (d, d).
    """
    n, _ = X.shape
    sigmoid = 1 - 1 / (1 + np.exp(X @ beta_hat))  # Predicted probabilities
    weights = np.diag(sigmoid * (1 - sigmoid) / n)  # Diagonal weight matrix
    hess_inv = np.linalg.inv(X.T @ weights @ X)  # Inverse of the Hessian matrix
    grad = X * (sigmoid[:, None]) - X * Y[:, None]  # Gradient of the log-likelihood
    mean = np.mean(grad, axis=0)
    term = grad - mean[None, :]  # Centering the gradient
    cov = term.T @ term / (n - 1)  # Covariance matrix of the gradient
    return hess_inv @ cov @ hess_inv / n  # Sandwich covariance matrix


def logistic_bootstrap_cov(B, X, Y):
    """
    Compute the bootstrap covariance matrix for logistic regression.

    The bootstrap covariance matrix is computed by resampling the data B times, 
    fitting the model to each resampled dataset, and calculating the covariance of 
    the resulting estimated coefficients.

    Parameters:
    B (int): The number of bootstrap resamples.
    X (numpy.ndarray): The design matrix of shape (n, d), where n is the number of observations 
                        and d is the number of features.
    Y (numpy.ndarray): The target vector of shape (n,), representing the binary observed values (0 or 1).

    Returns:
    numpy.ndarray: The bootstrap covariance matrix of shape (d, d).
    """
    n, _ = X.shape
    
    def bootstrap_sample():
        """
        Generate a bootstrap sample, fit the model, and return the estimated coefficients.
        """
        idx = np.random.choice(n, n, replace=True)  # Resample with replacement
        X_resampled, Y_resampled = X[idx], Y[idx]  # Resampled design matrix and target vector
        log_reg = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        log_reg.fit(X_resampled, Y_resampled)  # Fit logistic regression
        return log_reg.coef_[0]  # Return the estimated coefficients
    
    # Parallel computation of the resampled beta coefficients
    beta_bootstrap = np.array(Parallel(n_jobs=-1)(delayed(bootstrap_sample)() for _ in range(B)))
    
    beta_mean = np.mean(beta_bootstrap, axis=0)  # Mean of the resampled coefficients
    diff_resampled = beta_bootstrap - beta_mean  # Difference between each resampled coefficient and the mean
    return (diff_resampled.T @ diff_resampled) / B  # Bootstrap covariance matrix
