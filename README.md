# Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm

### Purpose
This repository contains all the data, functions, scripts to run simulations and analysis, and scripts to generate plots for the paper "Online Statistical Inference in Federated Learning with Differential Privacy".

### Pre-requisites 
1. Installation of R (Version 4.4.2) 
2. RStudio

### Dependencies
- MASS
- dplyr
- openxlsx

### Installation
#### Environment Requirements
Make sure you have the R environment and corresponding packages installed on your computer. The package and downstream analysis and simulations depend on several packages, including `MASS`, `dplyr`, etc. You can install the necessary packages using the following command:
```R
install.packages("MASS")
```
#### Steps for Execution  
Open R or RStudio and load the corresponding script file. You can run the script with the following command:
```R
source("localSGD_LinR.R")
```
### Data
The two type datasets used in this article.The first one is payment_fraud_dataset(downloaded from [Kaggle](https://www.kaggle.com/)), using a  logistic regression to detect online payments fraud. The second one is generating dataset,using logistic regression and Linear model.

### Project Structure
```
project/
├── exp1-exp4/ 
├── other_metrics/ 
├── real_dataset/
└── README.md
```
Contains the core code on both the generated and real datasets, including LocalSGD, FedSGD, CSL and BFGS algorithms. Each script file implements the core logic of the corresponding algorithm and provides examples of related functions.

### Reproducing the results
The results can be reproduced by running in the command window or Rstudio.For example:
```R
Rscript FedBFGS_LinR.R
Rscript Local_LinR.R
Rscript FedBFGS_LogR.R
Rscript Local_LogR.R
```
The above two files implement the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm and the local SGD algorithm, using a linear model and logistic regression.

### Contributions -  
Contributions of any kind are welcome, including issue reports, feature requests, and code contributions.

### License
This project uses the MIT license, please refer to the LICENSE file for details.

