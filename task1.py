import numpy as np

def get_eigen(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    
    real_eigenvalues = np.real(eigenvalues)
    real_eigenvectors = np.real(eigenvectors)
    
    n = matrix.shape[0]
    for i in range(n):
        val = eigenvalues[i]
        vec = eigenvectors[:, i]
        
        A_v = np.dot(matrix, vec)
        lambda_v = val * vec
        
        is_close = np.allclose(A_v, lambda_v)
        print(f"Eigenvalue {i+1}: {val}")
        print(f"Is A*v = lam*v?  :  {'Yes' if (is_close) else 'No'}")

    return real_eigenvalues, real_eigenvectors

if __name__ == "__main__":
    A = np.array([[4, -2],
                  [1, 1]])
    
    eigenvalues, eigenvectors = get_eigen(A)
    print("Real Eigenvalues:", eigenvalues)
    print("Real Eigenvectors:\n", eigenvectors)