![LinReg_plot](plots/Coverage_Probability_vs._Sample_Size_in_Linear_Regression.png)
![LogReg_plot](plots/Coverage_Probability_vs._Sample_Size_in_Logistic_Regression.png)
![FaultyLinReg_plot](plots/Coverage_Probability_vs._Sample_Size_in_Faulty_Linear_Regression.png)

# Covariance Robustness in Regression

This project compares three parameter inference methods used for parameter estimation in **Linear Regression**, **Faulty Linear Regression**, and **Logistic Regression**. We implement and evaluate **Fisher Covariance**, **Sandwich Estimator**, and **Bootstrap Resampling** techniques, applying them to three different regression models. This work investigates how these methods perform across different sample sizes and models.

The project consists of multiple scripts that simulate data, perform model fitting, calculate covariances, and evaluate coverage probabilities. It also includes experiments to test robustness in regression models by comparing the coverage of β (the true parameter) against different covariance estimates.

The models used in this project are:

1. **Linear Regression** (LR)
2. **Faulty Linear Regression** (FLR)
3. **Logistic Regression** (LogReg)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/CovarianceRobustnessInRegression.git
   cd CovarianceRobustnessInRegression
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
CovarianceRobustnessInRegression/
│
├── models/                   # Contains the model code (LinearReg, LogisticReg)
│   ├── LinearReg.py           # Linear Regression model implementation
│   ├── LogisticReg.py         # Logistic Regression model implementation
│
├── covariance/                # Contains covariance estimation methods
│   ├── LinearModelCov.py      # Covariance methods for Linear Regression models
│   ├── LogisticModelCov.py    # Covariance methods for Logistic Regression models
│
├── utils/                     # Helper functions for plotting and simulation
│   ├── plotting.py            # Plotting functions
│   ├── ModelSimulation.py     # Model simulation functions
│
├── plots/                     # Plots generated by the simulations
│   ├── Coverage_Probability_vs_Sample_Size_in_Linear_Regression.png
│   ├── Coverage_Probability_vs_Sample_Size_in_Faulty_Linear_Regression.png
│   ├── Coverage_Probability_vs_Sample_Size_in_Logistic_Regression.png
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── main.py                    # Main script to run the simulations
```


### Model Scripts:

* **`LinearReg.py`**: Implements the Linear Regression model simulation.
* **`LogisticReg.py`**: Implements the Logistic Regression model simulation.
* **`FaultyLinReg.py`**: Simulates faulty linear regression for comparison.

### Covariance Methods:

* **`LinearModelCov.py`**: Contains covariance functions (Fisher, Sandwich, Bootstrap) for linear regression.
* **`LogisticModelCov.py`**: Contains covariance functions (Fisher, Sandwich, Bootstrap) for logistic regression.

### Utilities:

* **`plotting.py`**: Contains the function to plot the results and save the plots to a folder.
* **`ModelSimulation.py`**: Contains the function to run the simulations and calculate coverage probabilities.

### Main:

* **`main.py`**: The main script that runs the experiments and plots the results.

## How the Experiment Works

### 1. Data Generation:

For **Linear Regression**, the data is generated as follows:

$$
y_i = x_i^T \beta^\star + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

* $x_i \sim \mathcal{N}(0, I_d)$ (i.e., a standard normal distribution).
* $\beta^\star$ is sampled from a unit sphere $S^{d-1}$ in $\mathbb{R}^d$.

For **Logistic Regression**, the model is:

$$
P(Y = y | X = x) = \frac{e^{y \beta^T x}}{1 + e^{x^T \beta}}, \quad y \in \{0, 1\}
$$

* $X_i$ and $\beta^\star$ are sampled similarly to the linear regression model.

For **Faulty Linear Regression**, we modify the linear model by adding a term $(x^T_i \theta^\star)^2$ where $\theta^\star \sim \text{Uni}(S^{d-1})$.

### 2. Covariance Estimation:

We evaluate three different methods to estimate the covariance of the model's parameters:

* **Fisher Covariance**: Based on the classical Fisher information matrix.
* **Sandwich Estimator**: A robust estimator derived from asymptotic theory.
* **Bootstrap Resampling**: Uses resampling to estimate the covariance from bootstrap samples.

### 3. Coverage Probability:

We repeat the experiment for different sample sizes (50, 100, 200, 400) and calculate the fraction of times that the true parameter $\beta^\star$ falls within a 90% confidence set defined by the estimated covariance.

The confidence set is defined as:

$$
C_n := \\{ \beta \in \mathbb{R}^d \mid (\beta - \beta_{bn})^T \Sigma_n^{-1} (\beta - \beta_{bn}) \leq \chi^2_{d,1-\alpha} \\}
$$

where $\Sigma_n$ is the estimated covariance, $\beta_{bn}$ is the estimated parameter, and $\chi^2_{d,1-\alpha}$ is the quantile from the chi-squared distribution.

## Insights and Discussion:

* **Linear and Logistic Regression**: When the model assumptions are correct (Linear and Logistic Regression), all three covariance methods (Fisher, Sandwich, Bootstrap) show similar performance in terms of coverage probability. Fisher covariance, being the classical approach, works well but slightly underperforms at smaller sample sizes.
* **Faulty Linear Regression**: When the model assumptions are violated (Faulty Linear Regression), the Sandwich estimator proves to be more robust, as it adjusts for potential model misspecifications. The Bootstrap estimator also performs well, taking advantage of its non-parametric nature and flexibility in modeling errors.

In cases where we are unsure about the assumptions of the model, especially when dealing with **Faulty Linear Regression**, the **Sandwich estimator** and **Bootstrap estimator** are preferred for their robustness, while the **Fisher estimator** is optimal when we are confident in the model assumptions.

## Visualize the Results

The results of the experiments are plotted and saved in the `plots/` directory. Each plot corresponds to the coverage probability of each covariance method (Fisher, Sandwich, Bootstrap) for a specific model (Linear Regression, Faulty Linear Regression, Logistic Regression).

Example plot filenames:

* `Coverage_Probability_vs_Sample_Size_in_Linear_Regression.png`
* `Coverage_Probability_vs_Sample_Size_in_Faulty_Linear_Regression.png`
* `Coverage_Probability_vs_Sample_Size_in_Logistic_Regression.png`

These plots visualize how the coverage probabilities vary with sample size and the choice of covariance estimation method. By examining these plots, we can gain insights into the robustness of each covariance estimator under different model assumptions.

---

## Credit

This assignment was designed by **John Duchi** for the PhD-level class **STATS 315A: Statistical Learning**.

