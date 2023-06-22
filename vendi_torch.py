"""
Vendi score
Pytorch implementation
"""

import torch
import torch.nn.functional as F


def weight_K(K, p=None, n=None):
    """
    Compute weighted similarity matrix
      K (Tensor [N, N] or [D, D]): Symmetric similarity matrix
      p (Tensor [N]): Probability vector over N examples
      n (int): Number of examples of original data matrix, 
               especially important if similarity matrix is [D, D]
    Return weighted similarity matrix (Tensor [N, N] or [D, D])
    """
    if p is None:  # Normalize by number of items
        if n is None:
            n = K.shape[0]
        return K / n
    else:
        assert K.shape[0] == p.shape[0]
        return K * torch.sqrt(torch.outer(p, p))

def weight_K_log(K, p=None, n=None):
    """
    Compute weighted similarity matrix in log space
      K (Tensor [N, N] or [D, D]): Log symmetric similarity matrix
      p (Tensor [N]): Log probability vector over N examples
      n (int): Number of examples of original data matrix, 
               especially important if similarity matrix is [D, D]
    Return weighted log similarity matrix (Tensor [N, N] or [D, D])
    """
    if p is None:  # Normalize by number of items
        if n is None:
            n = K.shape[0]
        return K - n  # Divide by number of items (in log space)
    else:
        assert K.shape[0] == p.shape[0]
        p_matrix = (p * .5).unsqueeze(0).expand(len(p), -1)  # Take square root (in log space) and expand view to [N x N]
        return K + p_matrix + p_matrix.T  # Multiply by outer product (in log space)

def normalize_K(K):
    d = torch.sqrt(torch.diagonal(K))
    return K / torch.outer(d, d)

def entropy_q(eigv, q=1):
    """
    Compute Shannon entropy over eigenvalues
      eigv (Tensor [N]): Eigenvalues of similarity matrix
      q (int): Sensitivity to minority examples, the higher the less sensitive
    Return Shannon entropy (Tensor)
    """
    eigv_ = eigv[eigv > 0]  # Mask to exclude zero eigenvalues
    if q == 1:
        return -torch.sum(eigv_ * torch.log(eigv_))
    if q == "inf":
        return -torch.log(torch.max(eigv))
    return torch.log(torch.sum(eigv_ ** q)) / (1 - q)

def score_K(K, q=1, p=None, normalize=False, p_log=False, n=None):
    """
    Compute Vendi score from similarity matrix
      K (Tensor [N, N] or [D, D]): Symmetric similarity matrix
      q (int): Sensitivity to minority examples, the higher the less sensitive
      p (Tensor [N]): Probability vector over N examples
      normalize (bool): Normalize similarity matrix
      p_log (bool): Calculate in log space
      n (int): Number of examples of original data matrix, 
               especially important if similarity matrix is [D, D]
    Return Vendi score (Tensor)
    """
    if normalize:
        K = normalize_K(K)

    if p_log:  # Probabilities are expected to be in log space
        K_ = torch.exp(weight_K_log(torch.log(K), p, n=n))
    else:
        K_ = weight_K(K, p, n=n)

    eigv = torch.linalg.eigvalsh(K_)
    return torch.exp(entropy_q(eigv, q=q))

def similarity_K(X, normalize=True, small=True):
    """
    Compute similarity matrix
      X (Tensor [N, D]): Data matrix, N examples (rows) in D dimensions (columns)
      normalize (bool): Normalize data matrix
      small (bool): Computer smaller similarity matrix [D, D], much quicker if D << N
    Return similarity matrix K (Tensor [N, N] or [D, D])
    """
    if normalize:
        X = F.normalize(X)  # Euclidean norm over vectors along dimension 1
        
    if small:  # Compute smaller similarity matrix [D, D]
        K = X.T @ X  # Covariance matrix
    else:  # Compute full [N, N]
        K = X @ X.T  # Gramian matrix
    return K

def score_X(X, q=1, p=None, normalize=True):
    """
    Compute Vendi score from data matrix
      X (Tensor [N, D]): Data matrix, N examples (rows) in D dimensions (columns)
      q (int): Sensitivity to minority examples, the higher the less sensitive
      p (Tensor [N]): Probability vector over N examples
      normalize (bool): Normalize data matrix
    Return Vendi score (Tensor)
    """
    n = X.shape[0]  # Number of examples (rows)
    small = p is None  # If no probabilities, compute smaller (and quicker) similarity matrix
    K = similarity_K(X, normalize=normalize, small=small)
    return score_K(K, q=q, p=p, n=n)
