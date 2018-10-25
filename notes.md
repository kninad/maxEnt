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
  the compendium + todo sections

- For the report, target about 2 pages of writing for a week. At least try to 
  document everything properly in this document. It need not be the final 
  version! Just add the rough notes here or document the writeups in ipynbs.

--------------------------------------------------------------------------------


## TODO: 25 OCT
- Download pickle files -- red + full
- Highest scoring pairs in full and reduced once again
- Sanity checks ??  + Plot codes -- high memory usage
- Approximate inf: Haas paper -- summarize and code it up


## ToDo

- All the sanity checks compiled into a ipython notebook

- Better documented code

- Run the maxent on the entire dataset -- make it more scalable

- Request swarm-server access from Peter. Just a single node will be enough.

- Review top-K calculation

- Do all of these computations in a jupyter notebook so as to separate
  the code from the exploratory, validation work.
  

# Compendium of discussions and notes

-   Data filtering: remove redundant features
    -   either appearing in all or
    -   use thresholds: gt= 0.99 or lt= 0.01

-   Top $k$ feature pair extraction using normalized L-measure
    -   some better guided way to accomplish that? check other exact and 
        approximate methods
    -   top-K computation -- Since L(x,y) is a value between 0 and 1 (clamped),
        0 indicating independence and 1 indicating dependence, the paris of 
        (x,y) that we want are those nearing 1. First level of extraction is here.
        So for a given value of K, only extract the top-K pairs based on their
        L(x,y) values since this will guarantee atleast K exact feature value
        pairs. Now for those K (x,y) pairs select from them the top-K exact
        feature value pairs based on the sorted order
    
    -   Alternative is to get the top-K feature pairs and just get the maximum
        scoring exactt feature value pair for each of them -- will get total K.
        Helps in spreading out the constraints over a wider set of feat-pairs.
        Easier and faster to compute.


    -   cross-validation for K-val -- plot and see elbow (keep trying with a higher k)

-   Approximate partitioning (for top-k feats)
    -   if exact partitioning is taking long time, making the approximation more
       robust and generalize
    -   McCallum paper: piece-wise likelihood method.
    -   treat everything as independent,  tune the normalization constant

- Rank ordering the edges in the partitions
    - Define the feature graph in the usual way
    - Drop the lowest weighted edges between the nodes (nodes are the feature
      /column indices)
    - Weights are the L_hat(i,j) values between two feat. indices
    - Find the connected components (partitions)

-   Market basket analysis
    -   sets that have related condition (diabetes, hypertension)
    -   pivot tables?
    -   reference: ESL 14.2
    -   `apriori` function from `mlextend` lib
    -   bump-hunting?

-   Compute the transition probabilities between ages
    -   refer to the last page of note for details
    -   something about steady state of markov chain.

## Validation checks

-   Final output as a probability distribution
    -   sanity
    -   marginals and constraints should (approximately) equal the mle output

-   Exploratory work
    -   check all 9C2 pairs (0-1 pairs): compare maxent and empirical
    -   disease id: 49, 53, 98 + any 4th disease with this triplet
    -   which pairs being on lead to others being on (causality?)
    -   underlying principles of maxent based optimization
    -   what is being approximated and how?

- As we add more constraints, the empirical and maxent distributions
  should get closer (in kl-divergence?). In 50-51 data i.e combined
  one, keep adding more. Sort of a pay as you go approach.
 
- See what happens to the plot of the aggregated num-diseases.

- Another check for validation. Take a pair of diseases having a
  higher prevalence (like diabetes, hypertension). With them include a
  third disease say VAR which was not partitioned with either of them.
  Now take another pair d1,d2. Check whether Pr(dia, hyp, VAR) &gt;
  Pr(d1, d2, VAR).
  
  
## Open-ended questions
-   Data sampled: is it biased?
    -   how to account for the sampling procedure
    -   MIT work (mentioned by Peter)

-   Model Validation
    -   distance between distributions (KL-divergence?)
    -   check whether distance is converging on removing certain feature pairs


--------------------------------------------------------------------------------

# Discussions


### Fri 12 Oct -- Hari's office

Discussion:
- MBA: guiding the constraints for the top pairs (k=1,2,3)
    - Some diseases disappear when we go from k=1 to k>1
    - eg: asthama, upper resp disease etc.
    - mba heuristics and apriori algorithm. Upto k=3 it was exact computation
    - mba can give higher order constraints to the maxent optimization 

- Check in Excel sheet whether with upper-resp = 1, what diseases co-occur
  most frequently.
  
- Maybe also check with the maxent constraints, with $\delta(x_i , y_j)$ for 
  that pair and also the L-measure L(X,Y)
 
- From mba, k>3 have low support so enforcing them exactly may lead to 
  overfitting.

- Try to think about the relation between mba-analysis and the notion of 
  statistical dependence calculated between 2 rvs through L-measure.
 
- MBA only finds the pairs where there is a strong positive correlation while
  Lmeasure can find all possibilities. How to modify mba for that?  
  
  - Use the notion of bit-flips. For a particular disease, flip its bits while
    keeping others same and check whether any disease still has high prevelence
    with it i.e (d_i=1, d_j=1) pairs. But here d_i=1 actually refers to no 
    presence of the disease due to the bit flips. Maybe through this we can get
    relations other than just pos-pos
  
- Finally, comparison between mba and L-measure. Use only either of them and see
  the outputs (maybe via plots like previous validation)





TODO:
- Try running on the entire dataset and check time for L_dict calculation
- With a high K value
- Separate the feature extraction and the optimization calls. Feature extraction
    is taking time. So run and store the object which will be used by the 
    Optimizer class
- Using numba to speed up the computations? Or some direct vectorized numpy
    code? Explore options here. Check in the code which values can be cached
    and are repeated again and again across the loop calls.

Questions:
- Swarm server access (ask Peter). Just require one node, will be enough.
- Faster constraint calculation?
- Rank-ordering edges? -- thresholding the edges on their L(x,y) values
- McCal paper: how to approximate


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
    - Finding the connected components -- approximate instead of exact computn    

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

-------------------

\newpage 

# Report

## Notes for report

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

\newpage

# Introduction
Take from Hari's motivation note

# Related Work
# Method
# Experiments,
# Applications
# Conclusion

