# Decision Making

A package designed to model decision-making behaviors in mice from 2AFC task experiments using dynamical systems and simulated reinforcement learning agents.

### User Instructions
First, create a virtual environment (there are many ways to do so, I use [Anaconda](https://www.anaconda.com/products/individual) and will demonstrate with that):
```
conda create --name=decision-making python=3.10.4
```
Activate the environment:
```
conda activate decision-making
```
This package relies on the [ssm package](https://github.com/lindermanlab/ssm/tree/master/ssm) by the Linderman Lab. Follow installation instructions there upon activating your virtual environment. Once installation is complete, navigate to the directory you want to contain the repo code with `cd` and clone the repository:
```
git clone https://github.com/johnlyzhou/decision-making.git
```
Then, `cd` into the repository:
```
cd decision-making
```
Install the required packages:
```
pip install -r requirements.txt
```
Install local packages and set up directory structure:
```
pip install -e .
```
Launch Jupyter notebook and open the example test.ipynb (currently in progress) to see how to use this package!
```
jupyter notebook
```
