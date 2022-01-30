# Content

Repository covering the analysis of financial time series with persistent homology.

### Description

We aim to analyze the evolution of daily returns of four key US stock markets indices -- DowJones, Nasdaq, Russell2000, SP500 -- over the period from 1989 to 2016 using persistent homology. To do so, we will follow the approach proposed by Marian Gidea and Yuri Katz in "[Topological Data Analysis of Financial Time Series: Landscapes of Crashes](https://arxiv.org/pdf/1703.04385.pdf)."

This project aims to reproduce the paper's experiments and further explore some variants if possible.

#### Short summary of the paper

The paper proposes a Topological Data Analysis (TDA) method to extract topological features from a multivariate time series with values in $R^d$ ($d = 4$ here as we consider four stock market indices). Features are computed from data slices extracted from the original time series via a sliding time window of length $w$. Using Vietoris-Rips filtrations, a persistence diagram and then a persistence landscape is computeed for each data slice, also called point cloud. A landscape being yielded per time window, each is further proceded into a single real value, then recombined into a final time series. The paper highlights that current and future market behavior can be evidenced using that final time series.