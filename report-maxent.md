---
title: Maximum Entropy Sparse Data Fitting
author: Ninad Khargonkar
date: 1 July 2018
abstract:
    This document will act as report/notes for the maximum
    entropy project
---

# Project Plan

Create a pdf (using pdflatex) from the markdown file:
`pandoc report.md -o out-report.pdf`


## Things to do

- Get started with coding (python)

- Read reference papers

- Familiarize with the subject matter

- Think about extensions, start *writing* a report!


## Coding

- Start with a basic toy example

- Get the data structures working

- Work on discrete data for now

- Search for appropriate Python packages


## Features 

1. Feature/attribute selection into one module

2. Optimization/fitting part into another (which accepts 'important' features from part 1)


## Specifics

1. Entropy estimation module
    - check if results from `pyitlib` and `pyentropy` are the same.

- L-measure module (+ normalized L-measure)

- Top $k$ feature pair extraction using L-measure

- Partition feature set into `m` partitions. (Can be done later)

- Use those $k$ along with marginal constraints for the optimization
  method (LBFGS)



### Entropy estimation

- Initially discrete-discrete case using the shrinkage estimate (JS)
    - OPT: try out other entropy estimators and compare performance

- Dis-Cts and Cts-Cts case to be done later.


\newpage 

# Introduction


# Related Work


# Method


# Experiments


# Applications


# Conclusion


































