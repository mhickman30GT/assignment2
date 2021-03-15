import multiprocessing
import os
import contextlib
import io
import collections

import mlrose_hiive as mlrose
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# GLOBAL VARIABLES
RANDOM_SEED = 14
CORE_COUNT_PERCENTAGE = .75  # NOTE: Any increase past this and the comp is unusable


class Pool:
    """Class to run a pool of algorithms"""

    def __init__(self, algorithms, num_cores=4):
        """Constructor for Pool"""
        self.algorithms = algorithms
        self.num_cores = min(num_cores, len(algorithms))

    @staticmethod
    def _run_algorithm(algorithm):
        """Run algorithm (capture and store stdout)"""
        print(f"    Start : {algorithm.name}")
        string_io = io.StringIO()
        with contextlib.redirect_stdout(string_io):
            algorithm.run()
        algorithm.stdout = string_io.getvalue()
        print(f"    Stop  : {algorithm.name}")
        return algorithm

    def run(self):
        """Run algorithms in multiprocessing pool"""
        print(f"Running {len(self.algorithms)} algorithms with {self.num_cores} cores")
        pool = multiprocessing.Pool(processes=self.num_cores)
        self.algorithms = pool.map(self._run_algorithm, self.algorithms)
        pool.close()
        pool.join()


class Runner:
    """ Mini runners class to make pool work """

    def __init__(self, name, instance):
        """Constructor for Runner"""
        self.name = name
        self.instance = instance
        self.data = None
        self.curves = None

    def run(self):
        """ Run the optimizer """
        # Run the optimizers
        data, curves = self.instance.run()
        # Save the results
        self.data = data
        self.curves = curves


def generate_edges(number_of_nodes, max_connections_per_node):
    """ Generate edges for Knapsack problem """
    """ 
        Stolen from mlrose hiive generator in order to gain access to edge params
        https://github.com/hiive/mlrose/blob/master/mlrose_hiive/generators/max_k_color_generator.py   
    """
    np.random.seed(RANDOM_SEED)
    # all nodes have to be connected, somehow.
    node_connection_counts = 1 + np.random.randint(
        max_connections_per_node, size=number_of_nodes
    )
    node_connections = {}
    nodes = range(number_of_nodes)
    for n in nodes:
        all_other_valid_nodes = [
            o
            for o in nodes
            if (o != n and (o not in node_connections or n not in node_connections[o]))
        ]
        count = min(node_connection_counts[n], len(all_other_valid_nodes))
        other_nodes = sorted(
            np.random.choice(all_other_valid_nodes, count, replace=False)
        )
        node_connections[n] = [(n, o) for o in other_nodes]
    # check connectivity
    g = nx.Graph()
    g.add_edges_from([x for y in node_connections.values() for x in y])
    for n in nodes:
        cannot_reach = [
            (n, o) if n < o else (o, n)
            for o in nodes
            if o not in nx.bfs_tree(g, n).nodes()
        ]
        for s, f in cannot_reach:
            g.add_edge(s, f)
            check_reach = len(
                [
                    (n, o) if n < o else (o, n)
                    for o in nodes
                    if o not in nx.bfs_tree(g, n).nodes()
                ]
            )
            if check_reach == 0:
                break
    edges = [(s, f) for (s, f) in g.edges()]
    return edges


# Story of my life
def generate_problems(exp_name, problem, p_config, runtype, run_config, outdir):
    """ Generate problem runner class for optimizer requested """
    problem_class = None

    # Create a problem
    if problem == 'FourPeaks':
        # Generate its instance
        problem_inst = mlrose.DiscreteOpt(length=p_config["length"],
                                          fitness_fn=mlrose.FourPeaks(p_config["t_pct"]),
                                          maximize=p_config["maximize"],
                                          max_val=p_config["max_val"])
        name = "FourPeaks"
        # Generate its class
        problem_class = ProblemClass(exp_name, name, problem_inst, runtype, run_config, outdir)

    elif problem == 'Knapsack':
        # Generate weights and values based on length
        values = list()
        weights = list()
        for v in range(1, p_config["length"] + 1):
            values.append(p_config["values"][v])
        for v in range(1, p_config["length"] + 1):
            weights.append(p_config["weights"][v])

        # Generate its instance
        problem_inst = mlrose.KnapsackOpt(length=p_config["length"],
                                          weights=weights,
                                          values=values,
                                          max_weight_pct=p_config["max_weight_pct"],
                                          maximize=p_config["maximize"],
                                          max_val=p_config["max_val"])
        name = "Knapsack"
        # Generate its class
        problem_class = ProblemClass(exp_name, name, problem_inst, runtype, run_config, outdir)

    elif problem == 'MaxKColor':
        # Generate edges
        edges = generate_edges(p_config["nodes"], p_config["max_connections"])
        # Generate its instance
        problem_inst = mlrose.MaxKColorOpt(length=p_config["length"],
                                           edges=edges,
                                           maximize=p_config["maximize"],
                                           max_colors=p_config["max_colors"])
        name = "KColor"
        # Generate its class
        problem_class = ProblemClass(exp_name, name, problem_inst, runtype, run_config, outdir)

    return problem_class


def generate_optimizers(optimizer, exp_name, problem_inst, o_config, outdir):
    """ Generate optimizer instance from input """
    instance = None
    opt = None

    if optimizer == "RHC":
        instance = mlrose.RHCRunner(problem=problem_inst,
                                    experiment_name=exp_name,
                                    output_directory=outdir,
                                    seed=o_config["seed"],
                                    iteration_list=o_config["iteration_list"],
                                    max_attempts=o_config["max_attempts"],
                                    restart_list=o_config["restart_list"])
        opt = "Random Hill Climb"
    elif optimizer == "SA":
        instance = mlrose.SARunner(problem=problem_inst,
                                   experiment_name=exp_name,
                                   output_directory=outdir,
                                   seed=o_config["seed"],
                                   iteration_list=o_config["iteration_list"],
                                   max_attempts=o_config["max_attempts"],
                                   temperature_list=o_config["temperature_list"],
                                   decay_list=[mlrose.ExpDecay])
        opt = "Simulated Annealing"
    elif optimizer == "GA":
        instance = mlrose.GARunner(problem=problem_inst,
                                   experiment_name=exp_name,
                                   output_directory=outdir,
                                   seed=o_config["seed"],
                                   iteration_list=o_config["iteration_list"],
                                   max_attempts=o_config["max_attempts"],
                                   population_sizes=o_config["population_sizes"],
                                   mutation_rates=o_config["mutation_rates"])
        opt = "Genetic Algorithm"
    elif optimizer == "MIMIC":
        instance = mlrose.MIMICRunner(problem=problem_inst,
                                      experiment_name=exp_name,
                                      output_directory=outdir,
                                      seed=o_config["seed"],
                                      population_sizes=o_config["population_sizes"],
                                      iteration_list=o_config["iteration_list"],
                                      max_attempts=o_config["max_attempts"],
                                      keep_percent_list=o_config["keep_percent_list"],
                                      use_fast_mimic=o_config["use_fast_mimic"])
        opt = "MIMIC"

    return opt, instance


class ProblemClass:
    """ Class for problem optimizers """

    def __init__(self, name, title, prob_inst, run_type, run_config, outdir):
        # Base Problem variables
        self.name = name
        self.title = title
        self.problem = prob_inst
        self.runtype = run_type
        self.config = run_config
        self.out_dir = outdir
        self.opt_list = dict()
        self.curves = dict()
        self.data = dict()
        self.random_seed = RANDOM_SEED
        self.core_count = round(multiprocessing.cpu_count() * CORE_COUNT_PERCENTAGE)
        # For tuning runs
        self.tune_opt = None
        self.opt_title = None
        self.hyperparameter = None
        self.param_values = None
        self.best = tuple()
        # For problem tuning runs
        self.tune_results = dict()
        self.tune_value = None

    def process_run_params(self):
        """ Process variables for run """
        # If its plots, setup one run for each optimizer
        print(f"Processing inputs for {self.runtype}")

        if self.runtype == "plots":
            for opt, params in self.config.items():
                name, inst = generate_optimizers(opt, self.name, self.problem, params, self.out_dir)
                self.opt_list[name] = inst

        # If its tuning, setup multiple runs over hyper param values
        if self.runtype == "tuning":
            self.tune_opt = self.config["opt"]
            self.hyperparameter = self.config["hp"]
            self.param_values = self.config["val"]
            # Loop through and create instances
            for val in self.param_values:
                params = self.config["json"]
                params[self.hyperparameter] = val
                name, inst = generate_optimizers(self.tune_opt, self.name, self.problem, params, self.out_dir)
                self.opt_title = name
                self.opt_list[str(val)] = inst

        if self.runtype == "protune":
            self.tune_value = self.config["val"]
            for opt, params in self.config["opts"].items():
                name, inst = generate_optimizers(opt, self.name, self.problem, params, self.out_dir)
                self.opt_list[name] = inst

    def dump(self, exp, data, curves):
        """Write data to disk"""
        # Write curves and stats to CSV
        pd.DataFrame.from_dict(curves).to_csv(os.path.join(self.out_dir, f'{exp}_curves.csv'), index=False)
        pd.DataFrame.from_dict(data).to_csv(os.path.join(self.out_dir, f'{exp}_data.csv'), index=False)

    def run(self):
        """ Run the optimizer """
        print(f"Running {self.runtype} for {self.title}")
        runlist = list()

        # Create the runners for pool
        for name, inst in self.opt_list.items():
            str_name = f'{name}'
            run_class = Runner(str_name, inst)
            if self.runtype == "protune":
                run_class.run()
            runlist.append(run_class)

        # If not running protuning, run pool
        if self.runtype != "protune":
            # Init pool
            pool = Pool(runlist, self.core_count)
            # Run the pool
            pool.run()
            # Grab the results
            runlist = pool.algorithms

        # Loop through and process results
        for runner in runlist:
            # Save the results
            self.data[runner.name] = runner.data
            self.curves[runner.name] = runner.curves

            # If plotting, dump the data
            if self.runtype == "plots":
                self.dump(runner.name, runner.data, runner.curves)

        # Generate plots
        if self.runtype == "plots":
            self.fit_plots()
        # Generate plot and dump best
        elif self.runtype == "tuning":
            self.tune_plots()
        elif self.runtype == "protune":
            self.tune_result()

    def fit_plots(self):
        """ Plot fitness, function evaluation, and time curves """
        # Create the 3 curves
        fit_fig, fit_ax = plt.subplots()
        fit_ax.set_title(f"{self.title}: Fitness vs Iterations")
        fit_ax.set_xlabel('Iterations')
        fit_ax.set_ylabel('Fitness')

        fev_fig, fev_ax = plt.subplots()
        fev_ax.set_title(f"{self.title}: Fitness vs Function Evals")
        fev_ax.set_xlabel('Function Evaluations')
        fev_ax.set_ylabel('Fitness')

        time_fig, time_ax = plt.subplots()
        time_ax.set_title(f"{self.title}: Fitness vs Time")
        time_ax.set_xlabel('Time')
        time_ax.set_ylabel('Fitness')

        # Loop through optimizers and plot
        for opt, curves in self.curves.items():
            # Plot fitness over iterations
            x = curves["Iteration"]
            y = curves["Fitness"]
            fit_ax.plot(x, y, alpha=0.8, label=opt)

            # Plot feval over iterations
            y = curves["Fitness"]
            x = curves["FEvals"]
            fev_ax.plot(x, y, alpha=0.8, label=opt)

            # Plot time over iterations
            y = curves["Fitness"]
            x = curves["Time"]
            time_ax.plot(x, y, alpha=0.8, label=opt)

        # Format and output all graphs
        fit_ax.grid(True)
        fit_ax.legend(loc="best")
        fit_fig.savefig(os.path.join(self.out_dir, f"{self.title}_fitness_curve.png"))

        fev_ax.grid(True)
        fev_ax.legend(loc="best")
        fev_fig.savefig(os.path.join(self.out_dir, f"{self.title}_feval_curve.png"))

        time_ax.grid(True)
        time_ax.legend(loc="best")
        time_fig.savefig(os.path.join(self.out_dir, f"{self.title}_time_curve.png"))

    def tune_plots(self):
        """ Plot hyperparameter tuning plots """
        tune_df = pd.DataFrame()
        # Loop through results to create data frame
        # Grab the best point for each curve on iteration
        for val, curves in self.curves.items():
            # Find max and save hyperparam value
            max_dict = curves.iloc[curves["Fitness"].idxmax()].to_dict()
            if isinstance(val, collections.Sequence):
                res = ''.join(filter(lambda i: i.isdigit(), val))
                max_dict[self.hyperparameter] = res
            else:
                max_dict[self.hyperparameter] = val

            # Create or append to dataframe
            if not tune_df.empty:
                tune_df = tune_df.append(max_dict, ignore_index=True)
            else:
                tune_df = pd.DataFrame(max_dict, index=[0])

        # Now that you have the data, make dem plots
        # Yeah its repeat code, cant be bothered to merge them
        fit_fig, fit_ax = plt.subplots()
        fit_ax.set_title(f"{self.title}: Fitness vs {self.hyperparameter}")
        fit_ax.set_xlabel(self.hyperparameter)
        fit_ax.set_ylabel('Fitness')

        fev_fig, fev_ax = plt.subplots()
        fev_ax.set_title(f"{self.title}: Fitness Evals vs {self.hyperparameter}")
        fev_ax.set_xlabel(self.hyperparameter)
        fev_ax.set_ylabel('Function Evaluations')

        time_fig, time_ax = plt.subplots()
        time_ax.set_title(f"{self.title}: Time vs {self.hyperparameter}")
        time_ax.set_xlabel(self.hyperparameter)
        time_ax.set_ylabel('Time To Converge')

        # Plot fitness over hyper-parameter
        x = tune_df[self.hyperparameter]
        y = tune_df["Fitness"]
        fit_ax.plot(x, y, alpha=0.8)

        # Plot feval over hyper-parameter
        x = tune_df[self.hyperparameter]
        y = tune_df["FEvals"]
        fev_ax.plot(x, y, alpha=0.8)

        # Plot time over hyper-parameter
        x = tune_df[self.hyperparameter]
        y = tune_df["Time"]
        time_ax.plot(x, y, alpha=0.8)

        # Format and output all graphs
        fit_ax.grid(True)
        fit_fig.savefig(os.path.join(self.out_dir, f"{self.hyperparameter}_fitness_curve.png"))

        fev_ax.grid(True)
        fev_fig.savefig(os.path.join(self.out_dir, f"{self.hyperparameter}_feval_curve.png"))

        time_ax.grid(True)
        time_fig.savefig(os.path.join(self.out_dir, f"{self.hyperparameter}_time_curve.png"))

    def tune_result(self):
        """ Gather results from tuning for plots """
        # Loop through results to create data frame
        # Grab the best point for each curve on iteration
        for opt, curves in self.curves.items():
            # Find max and save hyperparam value
            self.tune_results[opt] = curves.iloc[curves["Fitness"].idxmax()].to_dict()
