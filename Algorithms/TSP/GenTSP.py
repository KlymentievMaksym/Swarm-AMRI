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
if "GenTSP.exe" not in where:
    if "Algorithms" in where:
        where = listdir("./Algorithms")
        if "GenTSP.exe" not in where:
            if "TSP" in where:
                where = listdir("./Algorithms/TSP")
                if "GenTSP.exe" in where:
                    path = "./Algorithms/TSP/GenTSP"
                else:
                    raise Exception("GenTSP not found")
            else:
                raise Exception("GenTSP not found")
        else:
            path = "./Algorithms/GenTSP"
    else:
        raise Exception("GenTSP not found")
else:
    path = "./GenTSP"


def GenTSP(cities, pop_size=10, iterations=100, child_size=20, mutation_probability=0.5, **kwargs):
    info = kwargs.get("info", False)
    if info:
        print("[GenTSP] Started...")
    request = {
        "cities": cities.tolist(),
        "pop_size": pop_size,
        "iterations": iterations,
        "child_size": child_size,
        "mutation_probability": mutation_probability,
        "every": kwargs.get("every", 1)
    }
    if info:
        print("[GenTSP] Request created...")

    # process = subprocess.run(
    #     [path],
    #     input=json.dumps(request),
    #     capture_output=True,
    #     text=True
    # )

    if info:
        print("[GenTSP] Open GenTSP.exe...")
    process = subprocess.Popen(
        [path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if info:
        print("[GenTSP] Request:", request)
        print("[GenTSP] Send request to GenTSP.exe...")
    process.stdin.write(json.dumps(request))
    process.stdin.close()

    if info:
        print("[GenTSP] Request sent to GenTSP.exe...")
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
        # print("Message:", response["Message"])
        # print("Best distance:", response["best_distance"])
        # print("Best route:", response["best_route"])
        # print(f"Received line: {line}")  # Process each line here

    return_code = process.wait()
    stderr_output = process.stderr.read()
    if return_code != 0:
        print(f"Process failed with error: {stderr_output}")

    if info:
        print("[GenTSP] Done!")
    Plot().plotTSP(best_distance, best_route, cities, **kwargs)


if __name__ == "__main__":
    pop_size = 50
    iterations = 100
    child_size = 50
    prob = 0.5
    cities = Circle(10)[:, :2]
    atcp = GenTSP(cities, pop_size, iterations, child_size, prob, show_plot_animation=True, every=1, interval=100, info=True)
