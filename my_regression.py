import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
    C = X.T @ X / (N - 1)
    
    # Calculate the eigenvalue decomposition
    # Note: Since covariance matrices are real and symmetric, the eigenvalues
    #  are all real valued
    U, S, VT = np.linalg.svd(X)
    eig_vals = S ** 2 / (N - 1)
    
    # Calculate the inverted eigenvalue square root diagonal matrix
    E12 = np.diag(eig_vals ** (-1 / 2))
    
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


def weighted_upsample(data, feature, n_samples=500, seed=123456, plot=True):
    """ Upsamples the data, weighted by the distribution of a given feature.
    
    This function attempts to balance the given dataset by upsampling based on
    an inverse relationship to the current distribution of a particular feature.
    Data points associated with frequently occuring insatances of a feature are
    less likely to be sampled, and vice versa. For example, if there are two
    categories, with one occuring 4 times and the other occurring once, the
    minority feature will be four times as likel to be sampled. The weights
    are recalculated after each sample is added to the dataset.
    
    This function currently only works with discretized or categorical features.
    There is no stopping condition, and the number of samples must be chosen.
    Therefore, if an insufficient number is chosen, the data set will not become
    balanced.
    
    Parameters
        data: DataFrame = the dataset to be upsampled (and/or balanced)
        feature: str = the feature whose distribution determines the weighting
        n_samples: int = the number of samples to add to the dataset
        seed: int = the random seed used to make the process repeatable
        plot: bool = True -> plots a histogram of the feature distribution
        
    Returns
        resampled_data: DataFrame = the upsampled/balanced dataset
    """
    
    counts = len(data[feature].value_counts())
    month_wts = np.zeros(counts)
    upsampled_data = data.copy()
    np.random.seed(seed)
    for i in range(n_samples):
        # Weight the sampling by the current distribution
        counts = upsampled_data[feature].value_counts()
        ind_sort = list(counts.index.values.astype(int))
        month_wts[ind_sort] = (1 / counts.values) / (1 / counts.values).sum()

        # Select a month to sample from and resample from that selection
        month_id = np.random.choice(12, p=month_wts)
        month_data = upsampled_data.loc[upsampled_data[feature] == month_id]
        new_sample = resample(month_data, n_samples=1)

        # Append the new sample to the existing data
        upsampled_data = pd.concat([upsampled_data, new_sample], ignore_index=True)

    if plot:
        sns.countplot(x=upsampled_data[feature])
        
    return upsampled_data


def resample_ensemble(major_class, minor_class, n_folds=5, shuffle=True, seed=123456):
    """ Splits unbalanced data into subsets (folds).
    
    In each fold, the minority class is fully represented, while the majority
    class (the one with high frequency feature occurrences) is split into folds.
    This is an ensemble because the model can then be trained on each individual
    subset, and the final answer can be the average of the individually trained
    models.
    
    Parameters
        major_class: nd.array (observations, features) = the data corresponding
            to the high frequency class(es)
        minor_class: nd.array (observations, features) = the data corresponding
            to the low frequency class(es)
        n_folds:int = the number of subsets to split the majority class into
        shuffle: bool = indicates whether the data should be shuffled
        seed: int = the random seed used to make the process repeatable
        
    Returns
        folds: list(tuples) = the X and y data for each resampled subset
    """
    
    # Shuffle the data to avoid bias
    if shuffle:
        major_class.sample(frac=1, random_state=seed)
    fold_sz = len(major_class) // n_folds
    folds = []
    for k in range(n_folds):
        major_fold = major_class.iloc[k * fold_sz:(k + 1) * fold_sz, :]
        comb_data = pd.concat([minor_class, major_fold], axis=0)
        this_fold = comb_data.iloc[:, :-1], comb_data.iloc[:, -1]
        folds.append(this_fold)
    
    return folds


def BinaryEncoder(data, col_ls):
    """ Encodes numericized categorical data as a set of binary columns.
    
    Categorical data needs to be encoded such that it is usable by ML algorithms.
    However, large numbers of categories in one-hot encoding leads to large
    feature dimensions. Therefore, this function uses binary encoding to reduce
    the space required to encode these features. Binary encoding avoids the issue
    of collisions, which occurs with other methods such as hash encoding.
    
    The length of the binary encoding required is calculated based on the number
    of unique categories for each column. The encoded columns are concatenated
    and returned.
    
    This function requires that the categorical data has already been converted
    to be represented by a range of ints. This will be updated in later versions
    of the function.
    
    Paramters
        data: DataFrame = all data
        col_ls: list = list of categorical column names to be encoded
        
    Returns
        enc_data: DataFrame = categorical data encoded columns
        n_enc_col: int = total number of encoded columns in output
    """
    
    # Initialize a label encoder for dealing with non-numericized categories
    le = LabelEncoder()
    
    n_enc_col = 0
    data_cp = data.copy()
    for i, col in enumerate(col_ls):
        # Convert category names to a range of integers
        if (data[col].dtype == 'object') or (data[col].dtype == 'category'):
            data_cp[col] = le.fit_transform(data[col])
        
        # Find the number of unique categories
        n_cats = len(data_cp[col].unique())
        
        # Determine the length of the binary encoding
        n_dig = len(format(data_cp[col].max(), 'b'))
        n_enc_col += n_dig
        
        # Calculate the encoding for each category
        bin_enc = np.zeros((len(data_cp), n_dig), dtype=int)
        for n in range(n_cats):
            inds = data_cp.loc[data_cp[col] == n].index
            bin_val = [int(x) for x in format(n, f'0{n_dig}b')]
            bin_enc[inds, :] = bin_val
            
        # Assign new column names
        col_names = []
        for d in range(n_dig):
            col_names.append(f'{col}_{d}')
        
        # Concatenate the encoded columns
        if i == 0:
            enc_data = pd.DataFrame(bin_enc, columns=col_names)
        else:
            enc_data = pd.concat([pd.DataFrame(bin_enc, columns=col_names), enc_data], axis=1)
        
    return enc_data, n_enc_col
