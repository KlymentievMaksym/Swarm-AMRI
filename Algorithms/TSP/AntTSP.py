import subprocess
import json
from os import listdir
# import numpy as np
from tqdm import tqdm

# from Circle import Circle
if __name__ == "__main__":
    from Plot import Plot
    from Circle import Circle
else:
    from .Plot import Plot

where = listdir(".")
if "AntTSP.exe" not in where:
    if "Algorithms" in where:
        where = listdir("./Algorithms")
        if "AntTSP.exe" not in where:
            if "TSP" in where:
                where = listdir("./Algorithms/TSP")
                if "AntTSP.exe" in where:
                    path = "./Algorithms/TSP/AntTSP"
                else:
                    raise Exception("AntTSP not found")
            else:
                raise Exception("AntTSP not found")
        else:
            path = "./Algorithms/AntTSP"
    else:
        raise Exception("AntTSP not found")
else:
    path = "./AntTSP"


def AntTSP(cities, pop_size=10, iterations=100, alpha=0.5, beta=0.5, rho=0.5, Q=100, **kwargs):
    info = kwargs.get("info", False)
    plot = kwargs.get("plot", True)
    if info:
        print("[AntTSP] Started...")
    request = {
        "cities": cities.tolist(),
        "pop_size": pop_size,
        "iterations": iterations,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "Q": Q,
        "every": kwargs.get("every", 1)
    }
    if info:
        print("[AntTSP] Request created...")

    # process = subprocess.run(
    #     [path],
    #     input=json.dumps(request),
    #     capture_output=True,
    #     text=True
    # )

    if info:
        print("[AntTSP] Open AntTSP.exe...")
    process = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if info:
        print("[AntTSP] Send request to AntTSP.exe...")
    process.stdin.write(json.dumps(request))
    process.stdin.close()

    if info:
        print("[AntTSP] Request sent to AntTSP.exe...")
    best_distance = []
    best_route = []
    for line in tqdm(
        iter(process.stdout.readline, ''),
        desc="Processing",
        unit="step",
        bar_format="{l_bar}{bar:40}{r_bar}",
        colour='cyan',
        total=iterations//kwargs.get("every", 1)
    ):
        line = line.rstrip('\n')  # Remove trailing newline
        response = json.loads(line)
        best_distance.append(response["best_distance"])
        best_route.append(response["best_route"])
        # print("Best distance:", response["best_distance"])
        # print("Best route:", response["best_route"])
        # print(f"Received line: {line}")  # Process each line here

    return_code = process.wait()
    stderr_output = process.stderr.read()
    if return_code != 0:
        print(f"Process failed with error: {stderr_output}")

    if info:
        print("[AntTSP] Done!")
    if plot:
        Plot().plotTSP(best_distance, best_route, cities, **kwargs)
    print("[AntTSP] Distance:", best_distance[-1])
    return best_distance, best_route


if __name__ == "__main__":
    pop_size = 10
    iterations = 200
    cities = Circle(10)[:, :2]
    atcp = AntTSP(cities, pop_size, iterations, 5, 5, show_plot_animation=True, every=1, interval=100)
