## Ordinary Least Squares (OLS) Regression from Scratch

### Objective:
Implemented the Ordinary Least Squares (OLS) regression algorithm from first principles using NumPy, avoiding reliance on high-level libraries like scikit-learn. The goal was to deepen understanding of linear regression mechanics, including matrix operations and numerical stability.

### Key Features:

- **Matrix Algebra by Hand:** Developed custom functions for matrix multiplication, transposition, and inversion, utilizing Gauss-Jordan elimination with partial pivoting to ensure numerical stability.
  
- **OLS Fitting Process:** Implemented the closed-form solution for OLS regression:

  \[
  \hat{\beta} = (X^T X)^{-1} X^T y
  \]

- **Model Evaluation:** Computed model diagnostics including residuals, Sum of Squared Errors (SSE), variance, and R² score to assess the goodness-of-fit and model accuracy.

- **Validation:** Cross-verified the results by comparing the custom implementation against scikit-learn’s `LinearRegression`, demonstrating strong alignment in both coefficient estimates and predictive performance.

### Skills Demonstrated:

- Applied linear algebra principles to statistical modeling.
- Performed comprehensive model evaluation using R², SSE, and variance metrics.
- Demonstrated comprehension in Python class design, modularization, and testing practices.

### Result:
Matched the performance of scikit-learn’s implementation. This outcome confirmed the accurate implementation of core regression principles.
