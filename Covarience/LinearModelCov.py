import numpy as np

def linear_fisher_cov(X, Y, beta_hat):
    """
    Compute the Fisher covariance matrix for linear regression.

    The Fisher covariance matrix is computed as the inverse of the Gram matrix 
    (X^T X), scaled by the residual variance.

    Parameters:
    X (numpy.ndarray): The design matrix of shape (n, d), where n is the number of observations 
                        and d is the number of features.
    Y (numpy.ndarray): The target vector of shape (n,), representing the observed values.
    beta_hat (numpy.ndarray): The estimated coefficients of the model of shape (d,).

    Returns:
    numpy.ndarray: The Fisher covariance matrix, which is of shape (d, d).
    """
    n, d = X.shape
    sigma_hat = np.linalg.norm(Y - X @ beta_hat) ** 2 / (n - d)  # Residual variance
    return sigma_hat * np.linalg.inv(X.T @ X)  # Fisher covariance matrix


def linear_sandwich_cov(X, Y, beta_hat):
    """
    Compute the Sandwich covariance matrix for linear regression.

    The Sandwich estimator is calculated using the residuals and the outer product of the 
    design matrix X. It accounts for heteroscedasticity and non-independence in the residuals.

    Parameters:
    X (numpy.ndarray): The design matrix of shape (n, d), where n is the number of observations 
                        and d is the number of features.
    Y (numpy.ndarray): The target vector of shape (n,), representing the observed values.
    beta_hat (numpy.ndarray): The estimated coefficients of the model of shape (d,).

    Returns:
    numpy.ndarray: The Sandwich covariance matrix of shape (d, d).
    """
    res = (Y - X @ beta_hat) ** 2  # Squared residuals
    outer_products = X[:, :, None] @ X[:, None, :]  # Outer products of design matrix columns
    middle = np.tensordot(res, outer_products, axes=([0], [0]))  # Weighted outer products
    xtx_inv = np.linalg.inv(X.T @ X)  # Inverse of the Gram matrix
    return xtx_inv @ middle @ xtx_inv  # Sandwich covariance matrix


def linear_bootstrap_cov(B, X, Y, beta_hat):
    """
    Compute the bootstrap covariance matrix for linear regression.

    The bootstrap covariance matrix is computed by resampling the data B times, 
    fitting the model to each resampled dataset, and calculating the covariance of 
    the resulting estimated coefficients.

    Parameters:
    B (int): The number of bootstrap resamples.
    X (numpy.ndarray): The design matrix of shape (n, d), where n is the number of observations 
                        and d is the number of features.
    Y (numpy.ndarray): The target vector of shape (n,), representing the observed values.
    beta_hat (numpy.ndarray): The estimated coefficients of the model of shape (d,).

    Returns:
    numpy.ndarray: The bootstrap covariance matrix of shape (d, d).
    """
    n, _ = X.shape
    idx = np.random.choice(n, (B, n), replace=True)  # Bootstrap resampling indices
    X_resampled = X[idx]  # Resampled design matrix of shape (B, n, d)
    Y_resampled = Y[idx]  # Resampled target vector of shape (B, n)
    
    # Compute the inverse of the resampled design matrix and the estimated coefficients
    inverse_stack = np.linalg.inv(X_resampled.transpose(0, 2, 1) @ X_resampled)
    beta_hat_resampled = inverse_stack @ X_resampled.transpose(0, 2, 1) @ Y_resampled[:, :, None]
    
    # Compute the difference between resampled beta_hat and the original beta_hat
    diff_resampled = beta_hat_resampled.squeeze(-1) - beta_hat[None, :]
    
    # Compute the covariance matrix as the mean of the outer products of the differences
    return np.mean(diff_resampled[:, :, None] @ diff_resampled[:, None, :], axis=0)
