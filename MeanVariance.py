#!/usr/bin/env python
# coding: utf-8

# ## Mean Variance Portfolio Optimization

# ### Minimum Variance Solution

# $$
# \frac{
#     \Sigma^{-1}
#     *
#     \begin{bmatrix}
#     1  \\
#     1  \\
#     1  \\
#     \end{bmatrix}
# }{
#     \begin{bmatrix}
#     1 & 1 & 1
#     \end{bmatrix}
#     *
#     \Sigma^{-1}
#     *
#     \begin{bmatrix}
#     1  \\
#     1  \\
#     1  \\
#     \end{bmatrix}
# }=
# \begin{bmatrix}
# W_{_{AAPL}}  \\
# W_{_{JMP}}  \\
# W_{_{DIS}}  \\
# \end{bmatrix}$$

# Where $\Sigma^{-1}$ is the inverse of the covariance matrix, and the matrix of 1s has n elements, where n is the number of assets.

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta

myPortfolio = ["AAPL", "JPM", "GE", "DIS"]
def MinVarWeights(tickers: list):
    startDate = str(date.today() - timedelta(days = 365)) #Take a date 1 year in the past
    endDate = str(date.today()) #Take today's date
    data = yf.download(myPortfolio, start = startDate , end = endDate , progress = False)['Adj Close'].pct_change().dropna() #Download stock change data
    cov = data.cov() #Find the covariance matrix
    inv = np.linalg.inv(cov) #Take the inverse of the covariance matrix
    ones = np.ones(len(myPortfolio)) #Create identity matrix of length n
    minvar = (np.dot(inv, ones))/(np.dot(np.dot(ones.T, inv), ones)) #Find the min variance
    return minvar
MinVarWeights(myPortfolio)

