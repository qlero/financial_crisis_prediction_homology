# Content

Repository covering the analysis of financial time series with persistent homology.

### Description

The goal of this project is to analyze the evolution of daily returns of four major US stock markets indices (DowJones, Nasdaq, Russell2000, SP500) over the period 1989 – 2016 using persistent homology.

We follow the approach proposed in "[Topological Data Analysis of Financial Time Series: Landscapes of Crashes](https://arxiv.org/pdf/1703.04385.pdf)" by Marian Gidea and Yuri Katz.

A classical approach in TDA to extract topological features from multivariate time-series with values in $R^d$ ($d = 4$ here, since we are considering the evolution of four indices) consists in using a sliding window of fixed length $w$ to generate a sequence of $w$ points in $R^d$.

Using the Vietoris-Rips filtration, the persistence diagram of each of these point clouds is then computed and used as a topological feature for further analysis or processing of the initial data. This project aims at reproducing the experiments in the paper cited above and explore and discuss a few variants.
