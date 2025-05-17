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


def func(A, X, Y):
    return np.mean((Y - X @ A)**2)


def L2_reg(A, X, Y, lambda_):
    return np.mean((Y - X @ A)**2) + lambda_ * np.mean(A**2)


def L1_reg(A, X, Y, lambda_):
    return np.mean((Y - X @ A)**2) + lambda_ * np.mean(np.abs(A))


def elastic(A, X, Y, lambda_1, lambda_2):
    return np.mean((Y - X @ A)**2) + lambda_1 * np.mean(A**2) + lambda_2 * np.mean(np.abs(A))


def plot(de_err, pso_err, a_err, de, pso, save=False):
    print("Error DE: ", de_err)
    print("Error PSO: ", pso_err)
    print("Error Analytic: ", a_err)

    plt.title(f"DE, PSO, Analytic:\n{de_err}, {pso_err}, {a_err}")
    plt.plot(de[-2], label=f"DE {de[0]}")
    plt.plot(pso[-2], label=f"PSO {pso[0]}")
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


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

    pop_size = 10
    iterations = 10
    lambda_1 = 55
    lambda_2 = 10

    a = np.linalg.inv(x_train.T @ x_train + lambda_1 * np.eye(x_train.shape[1])) @ x_train.T @ y_train
    a_L2 = np.linalg.inv(x_train.T @ x_train + lambda_1 * np.eye(x_train.shape[1])) @ x_train.T @ y_train

    train = [x_train, y_train]

    path = data_path.split(".")[1].split("/")[-1]
    path = "./Lab7/Images/" + path

    funcs = [func, L2_reg, L1_reg, elastic]
    parameters = [[*train], [*train, lambda_1], [*train, lambda_2], [*train, lambda_1, lambda_2]]

    for fun, param in zip(funcs, parameters):
        de = DE(pop_size, iterations, fun, a_limits, parameters_to_pass=param, more=True)
        pso = PSO(pop_size, iterations, [0, 4], [-.15, .15], fun, a_limits, parameters_to_pass=param, more=True)
        de_err = error(y_test, x_test @ de[1])
        pso_err = error(y_test, x_test @ pso[1])
        if param == train:
            a_err = error(y_test, x_test @ a)
        elif len(param) == 3 and param[-1] == lambda_1:
            a_err = error(y_test, x_test @ a_L2)
        else:
            a_err = None
        plot(de_err, pso_err, a_err, de, pso, save=f"{path}_1__{fun.__name__}.png")

    # de = DE(pop_size, iterations, L2_reg, a_limits, parameters_to_pass=train + [lambda_1], more=True)
    # pso = PSO(pop_size, iterations, [0, 4], [-.15, .15], L2_reg, a_limits, parameters_to_pass=train + [lambda_1], more=True)
    # de_err = error(y_test, x_test @ de[1])
    # pso_err = error(y_test, x_test @ pso[1])
    # a_err = error(y_test, x_test @ a_L2)
    # plot(de_err, pso_err, a_err, de, pso, save=f"{path}_{L2_reg.__name__}.png")

    # de = DE(pop_size, iterations, L1_reg, a_limits, parameters_to_pass=train + [lambda_2], more=True)
    # pso = PSO(pop_size, iterations, [0, 4], [-.15, .15], L1_reg, a_limits, parameters_to_pass=train + [lambda_2], more=True)
    # de_err = error(y_test, x_test @ de[1])
    # pso_err = error(y_test, x_test @ pso[1])
    # plot(de_err, pso_err, None, de, pso, save=f"{path}_{L1_reg.__name__}.png")

    # de = DE(pop_size, iterations, elastic, a_limits, parameters_to_pass=train + [lambda_1, lambda_2], more=True)
    # pso = PSO(pop_size, iterations, [0, 4], [-.15, .15], elastic, a_limits, parameters_to_pass=train + [lambda_1, lambda_2], more=True)
    # de_err = error(y_test, x_test @ de[1])
    # pso_err = error(y_test, x_test @ pso[1])
    # plot(de_err, pso_err, None, de, pso, save=f"{path}_{elastic.__name__}.png")
