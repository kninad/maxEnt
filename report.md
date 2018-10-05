---
title: Chronic Disease Modelling 
author: Ninad Khargonkar 
date: 1 Nov 2018 
abstract: 
    This  document will act as report/notes for the
    project chronic disease modelling. All the
    notes from the discussions will be compiled here
    so as to make the process of writing a report at
    end easier.
---

# Project Plan

- Class/oop code, Proper self-documenting code.
    - Activate the `maxent` python2 environment for running the code
    - Partition code better documented.
- Dealing with multi-valued discrete data and continuous data (new
   methods for these cases) + additions to data utils code file.
- Friday 1pm meeting in Peter's office.
- Try to work on report/code whenever there is an hour or two gaps
- Since its an I.S, treat its time commitment as equivalent to a
  course (RL)
- After each meeting a short summary of the points discussed that day should be
  written down with the date along with adding the important things to try in 
  the compendium section.



## Discussions

### Fri 28 Sep -- with Prof Hari B.

- Comments in code and try to make it follow the best practices
- data set discussion
- Any number > 0 in disease prevalence indicates the person has it.
  - so map any number > 0 to 1
  - the number actually denotes the number of visits to the clinic
- The disease codes are in the html page for meps data (bookmarked)
- Try maxent on the enitre thing and play around with the binary data for now
- Clean the data, write proper documentation for it and and do exploratory analysis

### Fri 21 Sep

- ToDo
 - KL div
 - Cluster analysis (for co-occurance, not progression)
 - Exploratory factor analysis
 - MBA -- another possible way to get constraints for maxent

- Scaling up the maxent approach
 - Approximate partitioning:
  - Task1 -- McCal paper, treat as independent and introduce dependence with a 
    single normalization constant.
  - Task2 -- Rank order edges in the graph and drop the lowest scoring ones.
    Weights of the edges are probably the correlation?

- Why does 1yr with k=5 looks similar to merge with k=15?
 - maxent assumption -- everything is independent
 - as we get closer to the maxent assumption, the better it will perform
 - in the merge dataset, there is more likelihood of having some 
   correlation between the features. 

- Read up on markov chains and steady states.
 - Have to get static case working properly first.
 - But it will be useful for working out the transitions.

### Fri 14 Sep

-   better documenting code. this is the first priority
    -   write down the descriptions of the variables
    -   write down relevant formulas in comment when coding up a
        function to perform its equivalent computations
-   recent papers: exploratory factor analysis for disease associations
    -   more of covariance matrices, pca
    -   pca for indicator variables
    -   multi-morbidity, health services -- search terms
-   review papers -- cluster analysis for grouping diseases
    -   market basket -- most frequently seen
    -   coming up with the constraints for maxent using these other
        techniques for finding associations.
    -   can use market-basket for this
-   Q)  bayesian with uniform (non-informative) prior. Is it equivalent
        to the maxent without any constraints? but could always add some
        constraints to maxent to impose some expert knowledge. highly
        dependent on prior knowledge or some lower level biological
        stochastic model

#### Rough notes

-   review top-K calculation

-   As we add more constraints, the empirical and maxent distributions
    should get closer (in kl-divergence?). In 50-51 data i.e combined
    one, keep adding more. Sort of a pay as you go approach.
-   See what happens to the plot of the aggregated num-diseases.

-   Another check for validation. Take a pair of diseases having a
    higher prevalence (like diabetes, hypertension). With them include a
    third disease say VAR which was not paritioned with either of them.
    Now take another pair d1,d2. Check whether Pr(dia, hyp, VAR) &gt;
    Pr(d1, d2, VAR).

-   Do all of these computations in a jupyter notebook so as to separate
    the code from the exploratory, validation work.


## Compendium of discussions (things to try!)

-   Final output as a probability distribution
    -   sanity
    -   marginals and constraints should (approximately) equal the mle
        output
-   Data filtering: remove redundant features
    -   either appearing in all or
    -   use thresholds: gt= 0.99 or lt= 0.01
-   Top $k$ feature pair extraction using normalized L-measure
    -   some better guided way to accomplish that?
    -   check other exact and approximate methods
    -   just keep on trying with a higher k
    -   plot and see elbow
-   Approximate partitioning (for top-k feats)
    -   if exact paritioning is taking long time
    -   making the approximation more robust and generalizable
    -   McCallum paper: piecewise liklihood method.
    -   treat everything as independent
    -   tune the normalizn constant
-   Data sampled: is it biased?
    -   how to account for the sampling procedure
    -   MIT work
-   Market basket analysis
    -   sets that have related condition (diabetes, hypertension)
    -   pivot tables?
    -   reference: ESL 14.2
    -   `apriori` function from `mlextend` lib
    -   bump-hunting?
-   Model Validation
    -   distance between distributions (KL-divergence?)
    -   check whether distance is converging on removing certain feature
        pairs
-   Exploratory work
    -   check all 9C2 pairs (0-1 pairs): compare maxent and empirical
    -   disease id: 49, 53, 98 + any 4th disease with this triplet
    -   which pairs being on lead to others being on (causality?)
    -   underlying principles of maxent based optimization
        -   what is being approximated and how?
-   Compute the transition probabilities between ages
    -   refer to the last page of note for details
    -   something about steady state of markov chain.

Notes for report
----------------

-   Dealing with data sparsity -- maxent approach

-   entropy estimation topics
    -   try out other entropy estimators and compare performance
-   use of L measure -- normalization, top covariate selection
    -   selecting the top feature pairs
-   Partitioning of features into compartments -- transitive closure and
    connected components in a graph
    -   approximate partitions
-   optimization, cvx, numerical methods
    -   iterative scaling vs lbfgsb
-   Market basket analysis to see relevant diseases

-   Experiments, Plots etc

-   Contribute to `pyitlib` library (bonus points!)

-   New possible insights into the problem

------------------------------------------------------------------------

\newpage 

Report
======

Include sections like:

-   Introduction: Take from Hari's motivation note.
-   Related Work
-   Method
-   Experiments,
-   Applications
-   Conclusion

