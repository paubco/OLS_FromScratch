## Ordinary Least Squares From Scratch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class OLSFromScratch:
    def __init__(self):
        self.beta = None

    def validate_data(self, dependent, independent):
        """
        Validates input data for least squares regression.

        Args:
            dependent: The dependent variable (y)
            independent: The independent variable (X)
        
        Returns:
           y, X
            
        Raises:
            ValueError: If validation fails
        """
        
        # Validate dependent variable (y)
        y = np.asarray(dependent, dtype=float) 
        if y.ndim == 1:   # Ensure 1D shape
            y = y.reshape(-1, 1)  #Convert 1D array to column vector
        
        # Validate independent variables (X)
        X = np.asarray(independent, dtype=float)
        if X.ndim == 1:   
            X = X.reshape(-1, 1)  #Convert 1D array to column vector

        # Check if y and X have the same number of observations
        if y.shape[0] != X.shape[0]:
            raise ValueError('Number of samples in X and y do not match')

        return y, X 
    
    def matrix_multiply(self, A, B):
        """
        Performs matrix multiplication of A and B manually.
        
        Args:
            A: Left matrix (n x k)
            B: Right matrix (k x m)

        Returns:
            Resulting matrix (n x m)
            
        Raises:
            ValueError: If matrix dimensions are not compatible
        """
        A = np.array(A)
        B = np.array(B)
        
        if A.shape[1] != B.shape[0]:
            raise ValueError("Matrix dimensions do not align for multiplication")

        result = np.zeros((A.shape[0], B.shape[1]))

        for i in range(A.shape[0]):         # Rows of A
            for j in range(B.shape[1]):     # Columns of B
                for k in range(A.shape[1]): # Columns of A / Rows of B
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def matrix_transpose(self, A):
        """
        Transposes the input matrix A.

        Args:
            A: 1 or 2D matrix

        Returns:
            Transposed matrix of shape (columns, rows)
        """
        A = np.array(A)
        
        if A.ndim == 1:
            A = A.reshape(-1, 1)
        
        result = np.zeros((A.shape[1], A.shape[0]))

        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                result[j][i] = A[i][j]

        return result
    

    def matrix_inverse(self, A):
        """
        Computes the inverse of a square matrix using Gauss-Jordan elimination.

        Args:
            A: A square NumPy array (n x n)

        Returns:
            Inverse of the matrix A

        Raises:
            ValueError: If the matrix is not square
            np.linalg.LinAlgError: If the matrix is singular (non-invertible)
        """
        A = np.array(A, dtype=float)
        n = A.shape[0]
        
        #Check propper shape
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square to compute its inverse")

        I = np.eye(n)            # Create Identity mateix
        AI = np.hstack([A, I])   # Create augmented matrix [A | I]

        # Forward elimination with partial pivoting
        for i in range(n):
                max_row = np.argmax(np.abs(AI[i:, i])) + i #Find row with max absolute value
                if AI[max_row, i] == 0: #C heck for singular matrices
                    raise np.linalg.LinAlgError("Matrix is singular and cannot be inverted")
                AI[[i, max_row]] = AI[[max_row, i]]  # Swap rows
                AI[i] = AI[i] / AI[i, i]  # Normalize pivot
        
        #Back substitution
        for j in range(n):
                if i != j:
                    AI[j] = AI[j] - AI[i] * AI[j, i]

        return AI[:, n:]
    def fit(self, y, X, verbose=False):
        """
        Fits the OLS model to the data by computing the Beta coefficients.

        Args:
            y: Dependent variable
            X: Independent variable(s)
            verbose (bool): If True, print intermediate debug info

        Returns:
            numpy.ndarray: Estimated coefficients (Beta)
        """
        y, X = self.validate_data(y, X)
        Xt = self.matrix_transpose(X)
        Gram = self.matrix_multiply(Xt, X)
        Second = self.matrix_multiply(Xt, y)
        self.beta = self.matrix_multiply(self.matrix_inverse(Gram), Second)

        if verbose:
            print(f"Beta size: {self.beta.size}")
            print("Beta coefficients:\n", self.beta)

        return self.beta
    def predict(self, X, y_true=None, verbose=False):
        """
        Predicts values using the fitted OLS model and optionally evaluates performance.

        Args:
            X (numpy.ndarray): Independent variables for prediction
            y_true (numpy.ndarray, optional): True values for evaluation
            verbose (bool): If True, print debug info

        Returns:
            dict: Dictionary with predictions, and optionally residuals, s2, s, and R²
        """
        X = np.asarray(X, dtype=float)
        if self.beta is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        y_hat = self.matrix_multiply(X, self.beta)
        y_hat_flat = y_hat.ravel()
        result = {"y_hat": y_hat_flat}

        if y_true is not None:
            y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)
            residuals = y_true - y_hat
            SSE = float(np.dot(residuals.T, residuals))

            n, p = X.shape
            df = n - p
            s2 = SSE / df if df > 0 else np.nan
            s = np.sqrt(s2)

            y_centered = y_true - np.mean(y_true, axis=0)
            SST = float(np.dot(y_centered.T, y_centered))
            R2 = 1 - (SSE / SST)

            if verbose:
                print("Residuals:\n", residuals)
                print("SSE:", SSE)
                print("SST:", SST)
                print("R²:", R2)

            result.update({
                "residuals": residuals,
                "s2": s2,
                "s": s,
                "R2": R2
            })

        return result

# Create synthetic linear data: y = 3x + 5
np.random.seed(399)
X = np.arange(10).reshape(-1, 1)
y = 3 * X + 5 + np.random.normal(0, 1, size=X.shape)  # Add small noise

#Complile OLSFromScratch
model = OLSFromScratch()
model.fit(y, X, verbose=True)
results = model.predict(X, y_true=y, verbose=True)

#Compile Sklearn Linearregression
sk_model = LinearRegression(fit_intercept=False)  # Important: we already added intercept manually
sk_model.fit(X, y)
sk_beta = sk_model.coef_.reshape(-1, 1)
y_hat_sk = sk_model.predict(X)
r2_sk = r2_score(y, y_hat_sk)

# Print key outputs
print("From Scratch Predicted values:", results['y_hat'])
print("Sklearn Predicted values:", y_hat_sk)

print("From Scratch R² score:", results['R2'])
print("Sklearn R² score:", r2_sk)

