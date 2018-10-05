---
title: Chronic Disease Modeling 
author: Ninad Khargonkar 
date: 1 Nov 2018 
abstract: 
    This  document will act as report/notes for the project chronic disease 
    modeling. All the notes from the discussions will be compiled here to make
    the process of writing a final report easier. Create a pdf file from the 
    markdown using pandoc (via pdflatex) by the following command 
    `pandoc report.md -o output.pdf`
---

# Plan

- Object oriented and proper self-documenting code
    - make code commenting compliant to auto-documenting programs like doxygen
- Activate the `maxent` python2 environment for running the code
- Friday 1.30 pm meeting in Peter's office.
- Try to work on report/code whenever there is an hour or two gaps
- Since its an I.S, treat its time commitment as equivalent to a course (RL)
- After each meeting a short summary of the points discussed that day should be
  written down with the date along with adding the important things to try in 
  the compendium section


## Discussions

### Fri 28 Sep -- with Hari

Self note: 

- See section 5.1 in "Black-box VI" paper by Ranganath et al for a quick
  overview on modeling longitudinal time-series data of patients and their 
  medical records (path-lab observations)

Data set discussion:

- Any number > 0 in disease prevalence indicates the person has it.
    - so map any number > 0 to 1
    - the number actually denotes the number of visits to the clinic
    - Do this in the data-loading step
- The disease codes are in the html page for meps data (bookmarked)
- Try maxent on the entire thing and play around with the binary data for now
- Clean the data, write proper documentation for it and do exploratory analysis


### Fri 21 Sep

- KL divergence between empirical and resulted plots?

- Data analysis
    - MBA -- another possible way to get constraints for maxent
    - Cluster analysis (for co-occurrence, not progression)
    - Exploratory factor analysis

- Scaling up the maxent approach
    - Approximate partitioning:
    - Task1 -- McCallum paper, treat as independent and introduce dependence 
      with a single normalization constant.
    - Task2 -- Rank order edges in the graph and drop the lowest scoring ones.
    Weights of the edges are probably the correlation?

- Why does 1yr with k=5 looks similar to merge with k=15?
    - maxent assumption -- everything is independent
    - as we get closer to the maxent assumption, the better it will perform
    - in the merge data set, there is more likelihood of having some 
    correlation between the features. 

- Read up on markov chains and steady states.
    - Have to get static case working properly first.
    - But it will be useful for working out the transitions.


### Fri 14 Sep

- Better documented code -- 1st priority
  - write down the descriptions of the variables
  - write down relevant formulas in comment when coding up a
    function to perform its equivalent computations

- Recent papers: exploratory factor analysis for disease associations
    - more of co-variance matrices, pca
    - pca for indicator variables
    - multi-morbidity, health services -- search terms

- Review papers -- cluster analysis for grouping diseases
    - market basket -- most frequently seen
    - coming up with the constraints for maxent using these other 
      techniques for finding associations.
    - can use market-basket for this

- Question: bayesian with uniform (non-informative) prior. Is it equivalent
  to the maxent without any constraints? but could always add some
  constraints to maxent to impose some expert knowledge. highly
  dependent on prior knowledge or some lower level biological
  stochastic model


### Rough notes

- Review top-K calculation

- As we add more constraints, the empirical and maxent distributions
  should get closer (in kl-divergence?). In 50-51 data i.e combined
  one, keep adding more. Sort of a pay as you go approach.

- See what happens to the plot of the aggregated num-diseases.

- Another check for validation. Take a pair of diseases having a
  higher prevalence (like diabetes, hypertension). With them include a
  third disease say VAR which was not partitioned with either of them.
  Now take another pair d1,d2. Check whether Pr(dia, hyp, VAR) &gt;
  Pr(d1, d2, VAR).

- Do all of these computations in a jupyter notebook so as to separate
  the code from the exploratory, validation work.


## Compendium of discussions (things to try!)

- Approximate partitioning

- Run the maxent on the entire dataset

- Clean up the top-k constraints code

- All the extra sanity checks compiled into a ipython notebook

-   Final output as a probability distribution
    -   sanity
    -   marginals and constraints should (approximately) equal the mle output

-   Data filtering: remove redundant features
    -   either appearing in all or
    -   use thresholds: gt= 0.99 or lt= 0.01

-   Top $k$ feature pair extraction using normalized L-measure
    -   some better guided way to accomplish that?
    -   check other exact and approximate methods
    -   just keep on trying with a higher k
    -   plot and see elbow

-   Approximate partitioning (for top-k feats)
    -   if exact partitioning is taking long time
    -   making the approximation more robust and generalize
    -   McCallum paper: piece-wise likelihood method.
    -   treat everything as independent
    -   tune the normalization constant

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
    -   check whether distance is converging on removing certain feature pairs

-   Exploratory work
    -   check all 9C2 pairs (0-1 pairs): compare maxent and empirical
    -   disease id: 49, 53, 98 + any 4th disease with this triplet
    -   which pairs being on lead to others being on (causality?)
    -   underlying principles of maxent based optimization
    -   what is being approximated and how?

-   Compute the transition probabilities between ages
    -   refer to the last page of note for details
    -   something about steady state of markov chain.

--------------------------------------------------------------------------------

\newpage 

# Report

## Notes for report

- For math use the tex file sent by Peter in one of the emails

-   Dealing with data sparsity -- maxent approach

-   entropy estimation topics
    -   try out other entropy estimators and compare performance
-   use of L measure -- normalization, top co-variate selection
    -   selecting the top feature pairs
-   Partitioning of features into compartments -- transitive closure and
    connected components in a graph
    -   approximate partitions
-   optimization, convex, numerical methods
    -   iterative scaling vs lbfgsb
-   Market basket analysis to see relevant diseases

-   Experiments, Plots etc

-   Contribute to `pyitlib` library (bonus points!)

-   New possible insights into the problem


## Sections

-   Introduction: Take from Hari's motivation note.
-   Related Work
-   Method
-   Experiments,
-   Applications
-   Conclusion

