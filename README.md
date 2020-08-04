# Mixture Models

A toolbox for inference of mixture models

## Credits

This project builds upon the great work of Matt Johnson and others: https://github.com/mattjj/pybasicbayes

It started as a mere restructuring of the code but has by now significantly diverged from the original implementation.

## Installation
 
 Easiest way is to create a conda environment using the provided yml file
    
    conda env create -f ilr.yml
    
 Then head to the cloned repository and execute
 
    pip install -e .
    
 ## Examples
 
 A toy example of fitting a sine wave
 
    python examples/ilr/toy/evaluate_sine.py
    
 A toy example of fitting a step function
 
    python examples/ilr/toy/evaluate_step.py