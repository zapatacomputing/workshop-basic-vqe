import json
import sys
import matplotlib.pyplot as plt
from matplotlib import cycler


def plot_results(results):
    for name, step in results.items():
        x = []
        y = []
        try:
            x = step["results"]["values"]
            y = step["results"]["results"]
        except KeyError:
            print(f"Unable to find results in {name}")
            continue
        finally:
            plot(x, y)


def plot(x, y):
    colors = cycler("color", ["#00b578", "#ff3863"])
    plt.rc(
        "axes",
        facecolor="#E6E6E6",
        edgecolor="none",
        axisbelow=True,
        grid=True,
        prop_cycle=colors,
    )
    plt.rc("grid", color="w", linestyle="solid")
    plt.rc("xtick", direction="out", color="gray")
    plt.rc("ytick", direction="out", color="gray")
    plt.rc("patch", edgecolor="#E6E6E6")
    plt.rc("lines", linewidth=2)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_file>")
        sys.exit(-1)
    file_name = sys.argv[1]
    print(f"Plotting results from {file_name}")
    with open(file_name) as f:
        results = json.load(f)
        plot_results(results)
