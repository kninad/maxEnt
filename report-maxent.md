---
title: Maximum Entropy Sparse Data Fitting
author: Ninad Khargonkar
date: 1 July 2018
abstract:
    This document will act as report/notes for the maximum
    entropy sparse data fitting project.
---

# Project Plan

## Coding

- Class/oop code, Proper self-documenting code.

- Dealing with multi-valued discrete data and continuous data (new methods for
  these cases) + additions to data utils code file.

- Partitioning the features

- calculating Z (normalization constant) -- for binary, its easy. tough for 
  cts/dld

- check if mutual info values are very close negative to zero?

- data filtering -- remove redundant features (either appearing in all or none 
  or use some kind of threshold)

- sanity checks -- marginals and constraints should equal the mle output

- friday 1pm meeting

### Ask

- I(X;Y) becoming negative (both in normalized and un-normlaized setting)
  - threshold below at 0

- lbfgsb initial theta values

- marginal indicators when mutli-valued discrete
    - since in binary its just 0-1, its easy!

- top K selection
  - Plot and see elbow


### Specifics

1. Entropy estimation module
    - check if results from `pyitlib` and `pyentropy` are the same.

- L-measure module, normalized L-measure

- Top $k$ feature pair extraction using normalized L-measure

- Partition feature set into `m` partitions.


### Entropy estimation

- Initially discrete-discrete case using the shrinkage estimate (JS)

- Dis+Cts and Cts-Cts case to be done later.


## Report notes

- dealing with data sparsity

- entropy estimation topics
  - try out other entropy estimators and compare performance

- use of L measure -- normalization, top covariate selection

- Partitioning of features into compartments

- optimization, cvx, numerical methods
  - iterative scaling vs lbfgsb

- experiments, *write* report

- extensions, prasanna -- work, contribute to `pyitlib`

**Actual report/notes starts from the next page.**


\newpage 


# Introduction


# Related Work


# Method


# Experiments


# Applications


# Conclusion
