---
title: Chronic Disease Modelling
author: Ninad Khargonkar
date: 1 Nov 2018
abstract:
    This document will act as report/notes for the maximum entropy sparse data 
    fitting application project for chronic disease modelling. All the notes 
    from the discussions will be compiled here so as to make the process of 
    writing a report at end easier. After each meeting, a short summary of the
    points discussed that day should be written down with the date along with
    adding the important things to try in the compendium section.
---

# Project Plan

- Class/oop code, Proper self-documenting code.

- Dealing with multi-valued discrete data and continuous data (new methods for
  these cases) + additions to data utils code file.

- Friday 1.30-2.30pm meeting in Peter's office.

- Try to work on report/code whenever there is an hour or two gap in day.
    - Since its an I.S treat its time commitment as equivalent to a course (RL)
    
## Discussions

### Fri 14 Sep 

- a
- b
- c


# Compendium of discussions (things to try!)

- Final output as a probability distribution
    - sanity
    - marginals and constraints should (approximately) equal the mle output

- Data filtering: remove redundant features 
    - either appearing in all or
    - use thresholds: >= 0.99 or <= 0.01 

- Top $k$ feature pair extraction using normalized L-measure
      - some better guided way to accomplish that?
       - check other exact and approximate methods
      - just keep on trying with a higher k
      - plot and see elbow

- Approximate partitioning (for top-k feats)
      - if exact paritioning is taking long time
      - making the approximation more robust and generalizable
      - McCallum paper: piecewise liklihood method. 
       - treat everything as independent
       - tune the normalizn constant
    
- Data sampled: is it biased?
      - how to account for the sampling procedure
      - MIT work

- Market basket analysis
      - sets that have related condition (diabetes, hypertension)
      - pivot tables?
      - reference: ESL 14.2
      - `apriori` function from `mlextend` lib
      - bump-hunting?

- Model Validation
      - distance between distributions (KL-divergence?)
      - check whether distance is converging on removing certain feature pairs

- Exploratory work
      - check all 9C2 pairs (0-1 pairs): compare maxent and empirical
      - disease id: 49, 53, 98 + any 4th disease with this triplet
      - which pairs being on lead to others being on (causality?)

- Compute the transition probabilities between ages
      - refer to the last page of note for details
      - something about steady state of markov chain.



## Notes for report 

- Dealing with data sparsity -- maxent approach

- entropy estimation topics
    - try out other entropy estimators and compare performance

- use of L measure -- normalization, top covariate selection
    - selecting the top feature pairs

- Partitioning of features into compartments -- transitive closure and
  connected components in a graph
    - approximate partitions
  
- optimization, cvx, numerical methods
    - iterative scaling vs lbfgsb

- Market basket analysis to see relevant diseases

- Experiments, Plots etc

- Contribute to `pyitlib` library (bonus points!)

- New possible insights into the problem

---------------------------

\newpage 

# Report

Include sections like:

- Introduction
- Related Work
- Method
- Experiments,
- Applications
- Conclusion





