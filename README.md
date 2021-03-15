# Steps to install and run code:
 1. Download the repo
 2. Create environment with yml
 conda env create -f environment.yml
 3. Activate it
 conda activate assignment2

To run:
python main.py 
 # Select model
 # Plots just generates the metric plots for each optimizer
 # Tuning takes a value input and iterates over the list in the json config
 # Protune does the same as tuning but for the problem parameters/size
 -r --runtype (pick one of: plots , tuning , protune)
 # Select optimization problem
 -p --prob (pick problem: FourPeaks, Knapsack, MaxKColor, NN)
 # Select optimizer if tuning
 -o --opt (pick optimizer: RHC, SA, GA, MIMIC)
 # Select hyper parameter to tune if tuning
 -v --value (pick hyperparameter: the exact name in json config of the HP)
 # Select name for experiment/output files
 -n --name (it will auto-generate a name if blank)
 
config.json holds all hyperparameters, enter/modify in values or range to 

Examples:
python main.py -r plots -p FourPeaks -n fp_plots_3

python main.py -r tuning -p Knapsack -o SA -v temperature_list -n ks_tuning_test1

python main.py -r protune -p FourPeaks -o GA -v t_pct -n fp_protuning_test1

python main.py -r tuning -p NN -o SA -v temperature -n nn_tuning_test1


CODE REFEENCES:

All models, data processing, and plotting code stolen from the api/examples/discussion pages here:
https://github.com/hiive/mlrose/tree/master/mlrose_hiive
https://scikit-learn.org/stable/



                      EXTENDED REFERENCES
Knapsack:
http://www.es.ele.tue.nl/education/5MC10/Solutions/knapsack.pdf
https://www.youtube.com/watch?v=MacVqujSXWE&ab_channel=Computerphile

KColor:
https://minds.wisconsin.edu/handle/1793/59158
