---
title: Maximum Entropy Sparse Data Fitting
author: Ninad Khargonkar
date: 1 July 2018
abstract:
    This document will act as report/notes for the maximum
    entropy project
geometry: margin=1in
indent: true
header-includes:
    \usepackage{palatino}
---

# Introduction

Fitting Distributions for Sparse Data Via Maximum Entropy.  
Create a pdf (using pdflatex) from the markdown file:
`pandoc report.md -o out-report.pdf`


# Things to do

- Get started with coding (python)

- Read reference papers

- Familiarize with the subject matter

- Think about extensions


# Coding

- Start with a basic toy example

- Get the data structures working

- Work on discrete data for now

- Search for appropriate Python packages


## Features 

1. Feature/attribute selection into one module

2. Optimization/fitting part into another (which accepts 'important' features from part 1)


# Plan

1. Entropy estimation module

- L-measure module (+ normalized L-measure)

- Top $k$ feature pair extraction using L-measure

- Use those $k$ along with marginal constraints for the optimization
  method (LBFGS)

## Entropy estimation

- Initially discrete-discrete case: 




































