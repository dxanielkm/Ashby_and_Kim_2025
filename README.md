This repository contains data and figure generation files for the manuscript: “Coevolutionary cycling in allele frequencies and the evolution of virulence”.

**Notes:**

All files generated and tested using g++-14 with C++14 standard on MacOS 15.1
Typical install time < 1 min 
Expected runtime for typical input < 60 min

**Instructions:**

Compile the .cpp code in terminal, then run the Jupyter Notebooks to generate the figures. Use compile statements provided in the top of the code description. 

**Files:**

Each figure has a folder associated with it:

- Mean-Host-Density - Source code needed to compile the figure demonstrating mean host density increase with increase in specificity
- Parameter-Sweep - Source code needed to compile figure for main parameter sweep
- Sensitivity-Analysis - Source code needed to compile the parameter sweep figure across main model parameters
- Time-Series - Source code needed to generate time-series plot for model with default parameters