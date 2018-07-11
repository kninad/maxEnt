# maxEnt

## Fitting Distributions for Sparse Data Via Maximum Entropy

Code is in Python2 as some of libraries used do not support Python3.

### Libraries used:

- `numpy`, `scipy` (for the L-BFGS-B optimization module)
- `pyitlib` for discrete random vairable entropy calculations

report-maxent.md is the markdown report file. Create a pdf (using pdflatex and pandoc) from the markdown file:

`pandoc report-maxent.md -o out-report.pdf`
