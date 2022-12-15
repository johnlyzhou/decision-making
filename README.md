# Decision Making

A package designed to differentiate model-free and model-based decision-making paradigms from mice performing 2AFC experiments.

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
Run an analysis with the run script and a config file specifying your experimental parameters (examples [here](https://github.com/johnlyzhou/decision-making/tree/main/configs)):
```
python run.py path_to_config.yaml
```
