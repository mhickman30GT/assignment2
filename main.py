import argparse
import datetime
import json
import os
import shutil
import multiprocessing
import time

import problem as pro
import pandas as pd
import matplotlib.pyplot as plt
import neuralnet as nn

# GLOBAL VARIABLES
TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PATH = os.path.dirname(os.path.realpath(__file__))
CNFG = os.path.join(os.path.join(PATH, "data"), "config.json")


def get_args():
    """ Process args from command line """
    parser = argparse.ArgumentParser()

    # Set type of run
    parser.add_argument(
        "-r", "--runtype",
    )

    # Set problem to run
    parser.add_argument(
        "-p", "--prob",
    )

    # Set optimizer to tune
    parser.add_argument(
        "-o", "--opt",
    )

    # Set hyper param to tune
    parser.add_argument(
        "-v", "--value",
    )

    # Set name of the experiment
    parser.add_argument(
        "-n", "--name", default=f'RUN_DATA_{TIME}',
    )
    return parser.parse_args()


def plot_curves(title, rhc, sa, ga, mimic, hyperparameter, outdir):
    """ Plot tuning curves passed in """
    # Now that you have the data, make dem plots with repeated code
    fit_fig, fit_ax = plt.subplots()
    fit_ax.set_title(f"{title}: Fitness vs {hyperparameter}")
    fit_ax.set_xlabel(hyperparameter)
    fit_ax.set_ylabel('Fitness')

    fev_fig, fev_ax = plt.subplots()
    fev_ax.set_title(f"{title}: Fitness Evals vs {hyperparameter}")
    fev_ax.set_xlabel(hyperparameter)
    fev_ax.set_ylabel('Function Evaluations')

    time_fig, time_ax = plt.subplots()
    time_ax.set_title(f"{title}: Time vs {hyperparameter}")
    time_ax.set_xlabel(hyperparameter)
    time_ax.set_ylabel('Time To Converge')

    # Plot fitness over hyper-parameter
    x = rhc[hyperparameter]
    y = rhc["Fitness"]
    fit_ax.plot(x, y, alpha=0.8, label="RHC")

    # Plot feval over hyper-parameter
    x = rhc[hyperparameter]
    y = rhc["FEvals"]
    fev_ax.plot(x, y, alpha=0.8, label="RHC")

    # Plot time over hyper-parameter
    x = rhc[hyperparameter]
    y = rhc["Time"]
    time_ax.plot(x, y, alpha=0.8, label="RHC")

    # Plot fitness over hyper-parameter
    x = sa[hyperparameter]
    y = sa["Fitness"]
    fit_ax.plot(x, y, alpha=0.8, label="SA")

    # Plot feval over hyper-parameter
    x = sa[hyperparameter]
    y = sa["FEvals"]
    fev_ax.plot(x, y, alpha=0.8, label="SA")

    # Plot time over hyper-parameter
    x = sa[hyperparameter]
    y = sa["Time"]
    time_ax.plot(x, y, alpha=0.8, label="SA")

    # Plot fitness over hyper-parameter
    x = ga[hyperparameter]
    y = ga["Fitness"]
    fit_ax.plot(x, y, alpha=0.8, label="GA")

    # Plot feval over hyper-parameter
    x = ga[hyperparameter]
    y = ga["FEvals"]
    fev_ax.plot(x, y, alpha=0.8, label="GA")

    # Plot time over hyper-parameter
    x = ga[hyperparameter]
    y = ga["Time"]
    time_ax.plot(x, y, alpha=0.8, label="GA")

    # Plot fitness over hyper-parameter
    x = mimic[hyperparameter]
    y = mimic["Fitness"]
    fit_ax.plot(x, y, alpha=0.8, label="MIMIC")

    # Plot feval over hyper-parameter
    x = mimic[hyperparameter]
    y = mimic["FEvals"]
    fev_ax.plot(x, y, alpha=0.8, label="MIMIC")

    # Plot time over hyper-parameter
    x = mimic[hyperparameter]
    y = mimic["Time"]
    time_ax.plot(x, y, alpha=0.8, label="MIMIC")

    # Format and output all graphs
    fit_ax.grid(True)
    fit_ax.legend(loc="best")
    fit_fig.savefig(os.path.join(outdir, f"{hyperparameter}_fitness_curve.png"))

    fev_ax.grid(True)
    fev_ax.legend(loc="best")
    fev_fig.savefig(os.path.join(outdir, f"{hyperparameter}_feval_curve.png"))

    time_ax.grid(True)
    time_ax.legend(loc="best")
    time_fig.savefig(os.path.join(outdir, f"{hyperparameter}_time_curve.png"))

def plot_nn_curves(title, curve, hyperparameter, outdir):
    """ Plot tuning curves passed in """
    # Now that you have the data, make dem plots with repeated code
    fit_fig, fit_ax = plt.subplots()
    fit_ax.set_title(f"{title}: Accuracy vs {hyperparameter}")
    fit_ax.set_xlabel(hyperparameter)
    fit_ax.set_ylabel('Accuracy')

    l_fig, l_ax = plt.subplots()
    l_ax.set_title(f"{title}: Loss vs {hyperparameter}")
    l_ax.set_xlabel(hyperparameter)
    l_ax.set_ylabel('Loss')

    # Plot Train over hyper-parameter
    x = curve[hyperparameter]
    y = curve["Train Accuracy"]
    fit_ax.plot(x, y, alpha=0.8, label="Train")

    # Plot Test over hyper-parameter
    x = curve[hyperparameter]
    y = curve["Test Accuracy"]
    fit_ax.plot(x, y, alpha=0.8, label="Test")

    # Plot fitness over hyper-parameter
    x = curve[hyperparameter]
    y = curve["Loss"]
    l_ax.plot(x, y, alpha=0.8, label="Loss")

    # Format and output all graphs
    fit_ax.grid(True)
    fit_ax.legend(loc="best")
    fit_fig.savefig(os.path.join(outdir, f"{title}_{hyperparameter}_fitness_curve.png"))

    l_ax.grid(True)
    l_fig.savefig(os.path.join(outdir, f"{title}_{hyperparameter}_loss_curve.png"))

def find_optname(opt):
    """ Finds the neural net name for an optimizer """
    optname = "potato"
    if opt == "RHC":
        optname = "random_hill_climb"
    elif opt == "SA":
        optname = "simulated_annealing"
    elif opt == "GA":
        optname = "genetic_alg"
    return optname


def main():
    """ Main """
    # Process command line
    args = get_args()

    # Process directories and config
    dir_name = os.path.join(os.path.join(PATH, "out"), args.name)
    os.makedirs(dir_name)
    shutil.copy(CNFG, dir_name)

    # Open config
    with open(CNFG, "r") as open_file:
        config = json.load(open_file)

    # Create output directory
    out_dir = os.path.join(dir_name, args.name)
    os.makedirs(out_dir)

    # Find the problem's config
    p_config = config[args.prob]["params"]
    n_config = config[args.prob]

    # Create Run Config
    run_config = dict()

    # Core count for problem tuning
    core_count = round(multiprocessing.cpu_count() * .75)

    if args.prob != "NN":
        if args.runtype != "protune":
            # Set up tuning params if tuning
            if args.runtype == "tuning":
                run_config["hp"] = args.value
                run_config["opt"] = args.opt
                run_config["val"] = config[args.prob]["optimizers"][args.opt][args.value]
                run_config["json"] = config[args.prob]["optimizers"][args.opt]
            elif args.runtype == "plots":
                run_config = config[args.prob]["optimizers"]

            # Create problem class
            problem_class = pro.generate_problems(args.name, args.prob,
                                                  p_config, args.runtype,
                                                  run_config, out_dir)

            # Process run parameters
            problem_class.process_run_params()

            # Run problem class
            problem_class.run()

        else:
            # Run problem tuning
            problem_list = list()

            # Process args
            hyperparameter = args.value
            hyper_values = p_config[args.value]
            tune_config = p_config
            run_config["opts"] = config[args.prob]["optimizers"]

            for val in hyper_values:
                tune_config[hyperparameter] = val
                run_config["val"] = val

                # Create problem class
                problem_class = pro.generate_problems(args.name, args.prob,
                                                      tune_config, args.runtype,
                                                      run_config, out_dir)
                # Process run parameters
                problem_class.process_run_params()

                # Add to list for pool processing
                problem_list.append(problem_class)

            # Init pool
            pool = pro.Pool(problem_list, core_count)

            # Run the pool
            pool.run()

            rhc_curve = pd.DataFrame()
            sa_curve = pd.DataFrame()
            ga_curve = pd.DataFrame()
            mimic_curve = pd.DataFrame()

            # Gather results
            results = pool.algorithms

            for problem in results:
                for opt, curves in problem.tune_results.items():
                    # Find max and save hyperparam value
                    max_curve = curves
                    max_curve[hyperparameter] = problem.tune_value

                    # Place in correct curve
                    if opt == "Random Hill Climb":
                        if not rhc_curve.empty:
                            rhc_curve = rhc_curve.append(max_curve, ignore_index=True)
                            rhc_curve = pd.DataFrame(max_curve, index=[0])

                    elif opt == "Simulated Annealing":
                        if not sa_curve.empty:
                            sa_curve = sa_curve.append(max_curve, ignore_index=True)
                        else:
                            sa_curve = pd.DataFrame(max_curve, index=[0])

                    elif opt == "Genetic Algorithm":
                        if not ga_curve.empty:
                            ga_curve = ga_curve.append(max_curve, ignore_index=True)
                        else:
                            ga_curve = pd.DataFrame(max_curve, index=[0])

                    elif opt == "MIMIC":
                        if not mimic_curve.empty:
                            mimic_curve = mimic_curve.append(max_curve, ignore_index=True)
                        else:
                            mimic_curve = pd.DataFrame(max_curve, index=[0])

            # Plot problem tuning curves
            plot_curves(args.prob, rhc_curve, sa_curve,
                        ga_curve, mimic_curve,
                        hyperparameter, out_dir)
    else:
        # Neural Nets!
        # Process config params
        datafile = n_config["file"]
        filepath = os.path.join(os.path.join(PATH, "data"), datafile)
        nnlist = list()
        tunedict = dict()
        tune_curve = pd.DataFrame()
        hyperparameter = args.value
        optimizer = args.opt

        if args.runtype == "plots":
            for opt, params in n_config["optimizers"].items():
                optname = find_optname(opt)
                run_config["val"] = "potato"
                run_config["opt"] = optname
                run_config["params"] = params
                plotname = f'{opt}'
                nnclass = nn.generate_nets(plotname, filepath, p_config, run_config, out_dir)
                nnlist.append(nnclass)

        elif args.runtype == "tuning":
            # Tune NN hyperparams
            if optimizer == "NN":
                tunedict["RHC"] = list()
                tunedict["SA"] = list()
                tunedict["GA"] = list()
                hyper_values = p_config[args.value]
                tune_config = p_config
                for val in hyper_values:
                    tune_config[hyperparameter] = val
                    run_config["val"] = val
                    for opt, params in n_config["optimizers"].items():
                        plotname = f'{hyperparameter}{opt}{val}'
                        optname = find_optname(opt)
                        run_config["opt"] = optname
                        run_config["params"] = params
                        nnclass = nn.generate_nets(plotname, filepath, tune_config, run_config, out_dir)
                        tunedict[opt].append(nnclass)

            # Tune optimizer parameters
            else:
                hyper_values = n_config["optimizers"][optimizer][hyperparameter]
                tune_config = n_config["optimizers"][optimizer]
                optname = find_optname(optimizer)
                for val in hyper_values:
                    tune_config[hyperparameter] = val
                    run_config["val"] = val
                    run_config["opt"] = optname
                    run_config["params"] = tune_config
                    plotname = f'{hyperparameter}{val}'

                    # Create Neural Net Class
                    nnclass = nn.generate_nets(plotname, filepath, p_config, run_config, out_dir)
                    nnlist.append(nnclass)

        # Check for problem tuning
        if args.runtype == "tuning" and optimizer == "NN":
            # Loop through opts
            for opt, runlist in tunedict.items():
                # Init pool
                pool = pro.Pool(runlist, core_count)
                # Run the pool
                pool.run()
                # Save off results
                tunedict[opt] = pool.algorithms
        else:
            # Init pool
            pool = pro.Pool(nnlist, core_count)
            # Run the pool
            pool.run()
            # Save off results
            nnlist = pool.algorithms

        # Process results
        if args.runtype == "plots":
            # Generate LCA curves
            for net in nnlist:
                start_time = time.time()
                net.plot_lca()
                net.plot_loss()
                plot_time = round(time.time() - start_time,2)
                totaltime = net.runtime + plot_time
                print(f"{net.opt_type} Run time: {net.runtime}")
                print(f"{net.opt_type} PLot time: {plot_time}")
                print(f"{net.opt_type} Total time: {totaltime}")

        elif optimizer != "NN":
            # Generate tuning curves
            for net in nnlist:
                # Create data row
                res = dict()
                res[hyperparameter] = net.tune_value
                res["Train Accuracy"] = net.acc_train[0]
                res["Test Accuracy"] = net.acc_test[0]
                res["Loss"] = net.loss[0]
                # Add to dataframe
                if not tune_curve.empty:
                    tune_curve = tune_curve.append(res, ignore_index=True)
                else:
                    tune_curve = pd.DataFrame(res, index=[0])
            plot_nn_curves(optimizer, tune_curve, hyperparameter, out_dir)
        else:
            for opt, results in tunedict.items():
                tune_curve = pd.DataFrame()
                # Generate tuning curves
                for net in results:
                    # Create data row
                    res = dict()
                    res[hyperparameter] = f'{net.tune_value}'
                    res["Train Accuracy"] = net.acc_train[0]
                    res["Test Accuracy"] = net.acc_test[0]
                    res["Loss"] = net.loss[0]
                    # Add to dataframe
                    if not tune_curve.empty:
                        tune_curve = tune_curve.append(res, ignore_index=True)
                    else:
                        tune_curve = pd.DataFrame(res, index=[0])
                plot_nn_curves(opt, tune_curve, hyperparameter, out_dir)



if __name__ == "__main__":
    main()
