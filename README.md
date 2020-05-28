# Variational Item Response Theory (VIBO)

This repository contains PyTorch and Pyro code for "Variational Item Response Theory: Fast, Accurate, and Expressive" (https://arxiv.org/abs/2002.00276). The Pyro code contains fewer functional features than the PyTorch version but can be easily extended by the motivated reader. 

In this repository you can also find code for IRT with Hamiltonian Monte Carlo (HMC) and two purely deep neural baselines: MLE and Deep IRT (i.e. DKVMN-IRT). As in our experiments we also compared VIBO with Maximum Marginal Likelihood approaches (e.g. Expectation-Maximization); for this, we leverage the MIRT package.

NOTE: we are unable to release the Gradescope dataset publically. Please contact us if you require that data urgently. Otherwise, all other datasets used in the paper are supported here.

NOTE: we will be releasing a pip package supporting HMC and VIBO for IRT in Python. Please stay tuned!

## Abstract
Item Response Theory (IRT) is a ubiquitous model for understanding humans based on their responses to questions, used in fields as diverse as education, medicine and psychology. Large modern datasets offer opportunities to capture more nuances in human behavior, potentially improving test scoring and better informing public policy. Yet larger datasets pose a difficult speed / accuracy challenge to contemporary algorithms for fitting IRT models. We introduce a variational Bayesian inference algorithm for IRT, and show that it is fast and scaleable without sacrificing accuracy. Using this inference approach we then extend classic IRT with expressive Bayesian models of responses. Applying this method to five large-scale item response datasets from cognitive science and education yields higher log likelihoods and improvements in imputing missing data. The algorithm implementation is open-source, and easily usable.

## Setup Instructions
We use Python 3 and a Conda environment. Please follow the instructions below.

```
conda create -n vibo python=3 anaconda
conda activate vibo
conda install pytorch torchvision -c pytorch
pip install pyro-ppl
pip install tqdm nltk dotmap sklearn
```

The `config.py` file contains several useful global variables. Please change the paths there to be suitable to your own use cases.

## Downloading Data

I have included the real world data (with exception of Gradescope) in the public Google drive folder: https://drive.google.com/drive/folders/1ja9P5yzeUDyzzm748p5JObAEs_Evysgc?usp=sharing. Please unzip the folders in the `DATA_DIR` as specified by the config.

## How to Use

We will walk through a few commands for data processing and training models. First, this repository is setup as a package. Thus, for every fresh terminal, we need to run 
```
source init_env.sh
```
in order to add the correct paths.

### Create Simulation Data

To generate simulated data from an IRT model:
```
python src/simulate.py --irt-model 2pl --num-person 10000 --num-item 100 --ability-dim 1 
```
The generated data will be saved in a new folder inside `DATA_DIR`. We recommend using 1pl or 2pl unless the dataset size is rather large.

### Fitting MLE
To fit a Maximum Likelihood model, use the following command:
```
python src/torch_core/mle.py --irt-model 2pl --dataset 2pl_simulation --gpu-device 0 --cuda --num-person 10000 --num-item 100
```
This script has many command line arguments which the user should inspect carefully. If you do not have a CUDA device, remove the `--cuda` flag. 

If the dataset is a simulated one, it will read the `--num-person` and `--num-item` flags to know which sub-directory in `DATA_DIR` to load the data from. If the dataset is not simulated, those two flags are meaningless. 

If you wish to test missing data imputation, add the `--artificial-missing-perc` flag as we need to artificially hide some entries.

### Fitting VIBO
The commands for VIBO are quite similar to MLE. To run the **un-amortized** version, do:
```
python src/torch_core/vi.py --irt-model 2pl --dataset 2pl_simulation --gpu-device 0 --cuda --num-person 10000 --num-item 100
```

To run the **amortized** version, do:
```
python src/torch_core/vibo.py --irt-model 2pl --dataset 2pl_simulation --gpu-device 0 --cuda --num-person 10000 --num-item 100
```
Here, we highlight a few command line flags. First, `--conditional-posterior`, if present, conditions the approximate posterior over ability on the sampled items. Second, `--n-norm-flows` adds "Normalizing Flows" to the approximate posterior such that the resulting distribution need no longer be Gaussian but still reparameterizable (this is not mentioned in the main text but may be useful).

Several scripts are included to evaluate models. To get inferred latent variables, use `src/torch_core/infer.py`; to compute log marginal likelihoods, use `src/torch_core/marginal.py`; to analyze posterior predictives, use `src/torch_core/predictives.py`. 

### Fitting VIBO in Pyro
If you are more comfortable using a probabilistic programming language, we also include an implementation of VIBO in Pyro (which has been confirmed to return similar results to the PyTorch implementation). Do:
```
python src/pyro_core/vibo.py --irt-model 2pl --dataset 2pl_simulation --gpu-device 0 --cuda --num-person 10000 --num-item 100
```

### Fitting MCMC in Pyro
Pyro additionally makes it very easy to do inference with MCMC or HMC. We thus leverage it to compare VIBO to traditional methods of approximate Bayesian inference:
```
python src/pyro_core/hmc.py --irt-model 2pl --dataset 2pl_simulation --cuda --num-person 10000 --num-item 100
```
We emphasize that `--num-samples` and `--num-warmup` are important to getting good posterior samples. If you have several GPUs/CPUs, consider increasing `--num-chains` to greater than one.

### Fitting Deep IRT
Deep IRT is not a true inference model; rather it can only make predictions. Thus `--artificial-missing-perc` should be greater than 0. Use the command:
```
python src/dkvmn_irt/train.py --artificial-missing-perc 0.2 --num-person 10000 --num-item 100 --gpu-device 0 --cuda
```

### Tuning Parameters
In some settings, especially with small datasets, VIBO adds a KL regularization term that may impose too strong of a regularization. In practive, we find that adding a weight (less than 1) on the KL regularization terms helps to circumvent to this problem.
