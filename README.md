# Zappy
Repository of generalized functions, tools, and pipelines for processing intracranial EEG signals and electrical stimulation data.


## Installation
We recommend creating a new virtual environment using `Anaconda` or `virtualenv` before proceeding with this installation.
```
git clone https://github.com/akhambhati/zappy.git
cd zappy
pip install .
```

## Repository Organization
* __elstim__ - functions for processing electrical stimulation data
* __geometry__ - functions for processing spatial positioning information of electrodes
* __io__ - (deprecated) Pure python/numpy implementation of structures for handling labeled data/metadata 
* __sigproc__ - functions for signal processing, filtering, and time-series analysis
* __vis__ - basic visualization schemes to quickly plot signal time-series and heatmaps
* __pipelines__ - Combinations of above tools to run boilerplated pre-processing and analyses
