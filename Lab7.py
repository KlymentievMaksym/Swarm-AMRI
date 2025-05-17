import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from os import listdir

from Algorithms.Functions.DE import DE
from Algorithms.Functions.PSO import PSO

from sklearn.model_selection import train_test_split 

path = "."
lab_name = "Lab7"
if lab_name in listdir(path):
    path += "/" + lab_name
    if "Data" in listdir(path):
        path += "/Data"
        datas_path = [path + "/" + data for data in listdir(path)]
    else:
        raise Exception(f"Data directory in {lab_name} directory not found")
else:
    raise Exception(f"{lab_name} directory not found")


def error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def error_multiple(a, x_true, y_true):
    return np.mean((y_true - x_true @ a)**2)


def func(A, X, Y):
    return np.mean((Y - X @ A)**2)


def L2_reg(A, X, Y, lambda_):
    return np.mean((Y - X @ A)**2) + lambda_ * np.mean(A**2)


def L1_reg(A, X, Y, lambda_):
    return np.mean((Y - X @ A)**2) + lambda_ * np.mean(np.abs(A))


def elastic(A, X, Y, lambda_1, lambda_2):
    return np.mean((Y - X @ A)**2) + lambda_1 * np.mean(A**2) + lambda_2 * np.mean(np.abs(A))


def plot(de_err, pso_err, a_err, de, pso, every=1, save=False, animate=False):
    print("Error DE: ", de_err[-1])
    print("Error PSO: ", pso_err[-1])
    print("Error Analytic: ", a_err)

    fig, [ax1, ax2] = plt.subplots(2, figsize=(8, 10), sharey=True)

    de_min = ax1.plot(np.arange(len(de_err)) * every, de[-2], label=f"DE {de[0]}")[0]
    de_error = ax2.plot(np.arange(len(de_err)) * every, de_err, label=f"DE error {de_err[-1]}")[0]

    pso_min = ax1.plot(np.arange(len(de_err)) * every, pso[-2], label=f"PSO {pso[0]}")[0]
    pso_error = ax2.plot(np.arange(len(de_err)) * every, pso_err, label=f"PSO error {pso_err[-1]}")[0]

    ax1.set_title("Train Error")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness")

    ax2.set_title("Test Error")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fitness")

    ax1.grid(True)
    ax2.grid(True)

    if animate:
        def update(frame):
            plt.suptitle(f"Iteration {frame * every}")
            de_min.set_xdata(np.arange(frame) * every)
            de_min.set_ydata(de[-2][:frame])
            de_min.set_label(f"DE {de[-2][frame]}")

            de_error.set_xdata(np.arange(frame) * every)
            de_error.set_ydata(de_err[:frame])
            de_error.set_label(f"DE error {de_err[frame]}")

            pso_min.set_xdata(np.arange(frame) * every)
            pso_min.set_ydata(pso[-2][:frame])
            pso_min.set_label(f"PSO {pso[-2][frame]}")

            pso_error.set_xdata(np.arange(frame) * every)
            pso_error.set_ydata(pso_err[:frame])
            pso_error.set_label(f"PSO error {pso_err[frame]}")

            ax1.legend()
            ax2.legend()

            return de_min, de_error, pso_min, pso_error

        anim = animation.FuncAnimation(fig, update, frames=len(de_err), interval=100)
        if save:
            try:
                print(save)
                anim.save(save, writer="pillow", fps=20)
            except Exception as e:
                print(e)
        else:
            plt.show()
    else:
        if save:
            try:
                print(save)
                plt.savefig(save)
            except Exception as e:
                print(e)
            plt.close()
        else:
            plt.show()
    # plt.show()


for data_path in datas_path:
    try:
        data = pd.read_excel(data_path, sheet_name="Var10")
    except ValueError:
        data = pd.read_excel(data_path, sheet_name="boston_housing")
    dat = data.to_numpy()
    train, test = train_test_split(dat, test_size=0.25)

    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    x_train_ones = np.ones((x_train.shape[0], 1))
    x_test_ones = np.ones((x_test.shape[0], 1))

    x_train = np.concatenate((x_train_ones, x_train), axis=1)
    x_test = np.concatenate((x_test_ones, x_test), axis=1)

    a_limits = [[-1e2, 1e2]] * x_train.shape[1]

    pop_size = 100
    iterations = 1000
    lambda_1 = .55
    lambda_2 = .10
    every = 10

    a = np.linalg.inv(x_train.T @ x_train + lambda_1 * np.eye(x_train.shape[1])) @ x_train.T @ y_train
    a_L2 = np.linalg.inv(x_train.T @ x_train + lambda_1 * np.eye(x_train.shape[1])) @ x_train.T @ y_train

    train = [x_train, y_train]

    path = data_path.split(".")[1].split("/")[-1]
    path = "./Lab7/Images/" + path

    funcs = [func, L2_reg, L1_reg, elastic]
    parameters = [[*train], [*train, lambda_1], [*train, lambda_2], [*train, lambda_1, lambda_2]]

    for fun, param in zip(funcs, parameters):
        de = DE(pop_size, iterations, fun, a_limits, parameters_to_pass=param, more=True, every=every)
        pso = PSO(pop_size, iterations, [0, 4], [-.15, .15], fun, a_limits, parameters_to_pass=param, more=True, every=every)
        de_err = np.apply_along_axis(error_multiple, 1, de[-1], x_test, y_test)
        pso_err = np.apply_along_axis(error_multiple, 1, pso[-1], x_test, y_test)
        # de_err = error(y_test, x_test @ de[1])
        # pso_err = error(y_test, x_test @ pso[1])
        if param == train:
            a_err = error(y_test, x_test @ a)
        elif len(param) == 3 and param[-1] == lambda_1:
            a_err = error(y_test, x_test @ a_L2)
        else:
            a_err = None
        plot(de_err, pso_err, a_err, de, pso, every=every, animate=True) # , save=f"{path}__{fun.__name__}.gif"
