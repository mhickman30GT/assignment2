import multiprocessing
import os
import contextlib
import io
import collections

from sklearn import model_selection
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import mlrose_hiive as mlrose
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# GLOBAL VARIABLES
RANDOM_SEED = 14
CORE_COUNT_PERCENTAGE = .75  # NOTE: Any increase past this and the comp is unusable
ONE_HOT_ENCODING = True


class DataSet:
    """ Class holding values for dataset """

    def __init__(self, file):
        """ Constructor for Dataset """
        self.data_name = "Churn Data"
        self.file = file
        self.csv = pd.read_csv(self.file)
        self.label = "Exited"
        self.x_train = pd.DataFrame()
        self.x_test = pd.DataFrame()
        self.y_train = pd.Series()
        self.y_test = pd.Series()
        self.y_predict = pd.Series()

    def process(self):
        """ Processes data set """
        # Separate classification labels
        x = self.csv.drop(self.label, 1)
        y = self.csv[self.label]

        # Default to one hot for all sets
        if ONE_HOT_ENCODING:
            x = pd.get_dummies(x, columns=x.select_dtypes(include=[object]).columns)

        # Split into training and testing data
        self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(
            x, y, stratify=y, test_size=0.25, random_state=RANDOM_SEED)

        # Scale data using standardizer
        standardize = preprocessing.StandardScaler()
        self.x_train = standardize.fit_transform(self.x_train)
        self.x_test = standardize.transform(self.x_test)


def generate_nets(exp_name, file, p_config, run_config, outdir):
    """ Generate problem runner class for optimizer requested """
    # Generate the dataset instance
    dataset = DataSet(file)
    
    # Generate the instance
    problem_inst = mlrose.neural.NeuralNetwork(
                        hidden_nodes=p_config["hidden_layers"],
                        activation=p_config["activation"],
                        algorithm=run_config["opt"],
                        max_iters=run_config["params"]["max_iters"],
                        bias=True,
                        is_classifier=True,
                        learning_rate=p_config["learning_rate"],
                        early_stopping=True,
                        clip_max=p_config["clip_max"],
                        # Opt Tuning Params
                        restarts=run_config["params"]["restarts"],
                        schedule=mlrose.ExpDecay(run_config["params"]["temperature"]),
                        pop_size=run_config["params"]["pop_size"],
                        mutation_prob=run_config["params"]["mutation"],
                        max_attempts=run_config["params"]["max_attempts"],
                        random_state=RANDOM_SEED,
                        curve=True,
                        )
    name = "Neural_Network"

    # Generate its class
    problem_class = NNClass(exp_name, name, dataset, problem_inst, run_config, outdir)

    return problem_class


class NNClass:
    """ Class for optimizing neural networks """

    def __init__(self, name, title, data, prob_inst, run_config, outdir):
        # Base Problem variables
        self.name = name
        self.title = title
        self.dataset = data
        self.instance = prob_inst
        self.config = run_config
        self.out_dir = outdir
        self.opt_list = dict()
        self.curves = dict()
        self.random_seed = RANDOM_SEED
        self.core_count = round(multiprocessing.cpu_count() * CORE_COUNT_PERCENTAGE)
        # For tuning runs
        self.tune_results = dict()
        self.tune_value = self.config["val"]
        self.acc_train = None
        self.acc_test = None
        self.loss = None
        # For LCA
        self.train_sizes = None
        self.train_scores = None
        self.test_scores = None
        self.fit_times = None
        self.score_times = None

    def run(self):
        """ Run the neural net """
        print(f"Running {self.title}")

        # Pre-process data
        self.dataset.process()

        # Run fit and predict
        self.instance.fit(self.dataset.x_train, self.dataset.y_train)
        self.dataset.y_predict = self.instance.predict(self.dataset.x_test)

        # Generate scores
        self.acc_train = accuracy_score(self.dataset.y_train, self.instance.predict(self.dataset.x_train)),
        self.acc_test = accuracy_score(self.dataset.y_test, self.instance.predict(self.dataset.x_test)),
        self.loss = self.instance.loss,


    def plot_lca(self):
        """ Plots LCA curve """
        """ NOTE: DO NOT RUN INSIDE POOL """
        self.train_sizes, self.train_scores, self.test_scores, self.fit_times, self.score_times = model_selection.learning_curve(
            self.instance, self.dataset.x_train, self.dataset.y_train, n_jobs=self.core_count, return_times=True)

        # Generate figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        # Compute stats
        train_scores_mean = np.mean(self.train_scores, axis=1)
        train_scores_std = np.std(self.train_scores, axis=1)
        test_scores_mean = np.mean(self.test_scores, axis=1)
        test_scores_std = np.std(self.test_scores, axis=1)
        fit_times_mean = np.mean(self.fit_times, axis=1)
        fit_times_std = np.std(self.fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(
            self.train_sizes,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        axes[0].fill_between(
            self.train_sizes,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        axes[0].plot(self.train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        axes[0].plot(self.train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
        axes[0].legend(loc="best")
        axes[0].set_title(f"Learning Curve for Neural Network")
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Accuracy")
        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(self.train_sizes, fit_times_mean, "o-")
        axes[1].fill_between(
            self.train_sizes, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1
        )
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")
        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, "o-")
        axes[2].fill_between(
            fit_times_mean,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
        )
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")

        fig.savefig(os.path.join(self.out_dir, f"{self.name}_time_curve.png"))

    def plot_loss(self):
        """ Plot loss curve from NN """
        fig, axes = plt.subplots()

        plt.plot(self.instance.loss_curve_)
        plt.title(f"Loss Curve for Neural Networks")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        fig.savefig(os.path.join(self.out_dir, f"{self.name}_loss_curve.png"))
