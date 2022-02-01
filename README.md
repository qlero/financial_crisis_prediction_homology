# Overview

Repository covering the analysis of financial time series with persistent homology and its applicability to financial crisis prediction.

This repository is:

- First a reproduction of the Marian Gidea and Yuri Katz's results, proposed in "[Topological Data Analysis of Financial Time Series: Landscapes of Crashes](https://arxiv.org/pdf/1703.04385.pdf)," 491, 0378-4371, Physica A: Statistical Mechanics and its Applications, Elsevier BV, Feb. 2008.

- Secondly, an extension of some of its results along with further interpretations.

### Description

Using persistence homology, we analyze the evolution of daily returns of four key US stock markets indices -- DowJones, Nasdaq, Russell2000, SP500 -- over the period from 1989 to 2016.

#### Short summary of the basis paper

The paper proposes a Topological Data Analysis (TDA) method to extract topological features from a multivariate time series with values in $R^d$ ($d = 4$ here as we consider four stock market indices). Features are computed from data slices extracted from the original time series via a sliding time window of length $w$. Using Vietoris-Rips filtrations, a persistence diagram and then a persistence landscape is computeed for each data slice, also called point cloud. A landscape being yielded per time window, each is further proceded into a single real value, then recombined into a final time series. The paper highlights that current and future market behavior can be evidenced using that final time series.