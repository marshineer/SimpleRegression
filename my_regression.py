def whitening_transform(X, ZCA=False):
    """ Computes a transformation matrix to perform data whitening.
    
    Whitening both normalizes and decorrelates the data. The input array
    is in the form (instances x features). The whitened result will have the
    same dimensions and has been normalized to the range [-1, 1]. PCA-whitening
    is used with dimensionality reduction. ZCA-whitening rotates the data to 
    eliminate alignment with the principal components. It is used to keep the
    data as close as possible to its original form. 
    
    Note: The data must be centered before calculating the whitening transform.
    Whitening should only be performed on the test set, and then the same 
    transforms applied to the training set, to avoid overfitting.
    
    Arguments
        X: np.array (n x p) = centered data to be whitened
        ZCA: bool = True -> PCA whitening, False -> ZCA whitening
        
    Returns
        W: np.array (p x p) = whitening tranformation matrix
    """
    
    # Calculate the covariance matrix
    N = X.shape[0]
    # C = np.cov(X, rowvar=False, ddof=0)
    C = X.T @ X / (N - 1)
    # print(np.sum(C @ C.T - C.T @ C))
    
    # Calculate the eigenvalue decomposition
    # Note: Since covariance matrices are real and symmetric, the eigenvalues
    #  are all real valued
    # V, M = np.linalg.eigh(C)
    # print(np.sum((M[:, -1] * V[-1]) - (C @ M[:, -1])))
    # print((C * V[-1]) - (C @ M[:, -1])))
    U, S, VT = np.linalg.svd(X)
    eig_vals = S ** 2 / (N - 1)
    # print(eig_vals)
    # print(np.sum((VT.T[:, -1] * eig_vals[-1]) - (C @ VT.T[:, -1])))
    # print(np.sum((VT.T[:, 0] * eig_vals[0]) - (C @ VT.T[:, 0])))
    
    # Calculate the inverted eigenvalue square root diagonal matrix
    E12 = np.diag(eig_vals ** (-1 / 2))
    # print(E12.shape)
    
    # Calculate the whitening transformation matrix
    W = E12 @ VT
    if ZCA:
        W = VT.T @ W
    
    return W


def adjusted_R2(R2, N, p):
    """ Calculates the adjusted R^2 correlation.
    
    The adjusted R^2 value accounts for the number of features in a dataset. If
    additional features do not improve the prediction, the adjusted-R^2 value is
    penalized. 
    
    Reference: http://net-informations.com/ds/psa/adjusted.htm
    
    Arguments
        R2: scalar = the coefficient of determination
        N: scalar = number of instances (data points)
        p: scalar = number of features
        
    Returns
        adj_R2: scalar = adjusted-R^2 score
    """
    
    return 1 - (1 - R2) * (N - 1) / (N - p - 1)
    