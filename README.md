# ETGEMs
The  scripts for construction of enzymatic and thermodynamic constrained GEMs in a single Pyomo modelling framework.

## Environment

The scripts were written and tested with Python 3.5.5. 

The core libraries essential for the pipeline including: 

1) Cobrapy toolkit: version --0.13.3 (recommend, because the latest version has removed a required function of "convert_to_irreversible")；

2) Pyomo package: version --5.7；

3) Gurobi solver: version --9.0.2;

In addition，the Pandas and related packages. 

## Software

The packages used to run the pipeline was listed in requirements.txt. To install the requirements using pip, run the following code at command-line:

```shell
$ pip install -r requirements.txt
```

To create a stand-alone environment named ETGEMs with Python 3.5.5 and all the required package versions (especially for cobrapy is also available), run the following code:

```shell
$ conda create -n ETGEMs python=3.5.5
```
```shell
$ conda activate ETGEMs
```
```shell
$ pip install -r requirements.txt
```
```shell
$ python -m ipykernel install --user --name ETGEMs --display-name "ETGEMs"
```
  You can read more about using conda environments in the [Managing Environments](http://conda.pydata.org/docs/using/envs.html) section of the conda documentation. 

## Steps to reproduce the main analysis in the publication

Typical results can be reproduced by executing the Jupyter Python notebooks:

+ ETGEMs_function.py

  ——Pyomo template model and functions definition file used for following analysis：

  #####       Get_Figure1.ipynb

  #####       Get_Figure2.ipynb

  #####       Get_Table D.ipynb

  #####       Get_Table E.ipynb


