import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import math


def load_results_data(result_obj):
    # If the input is a string, return the results section directly
    if type(result_obj) == str:
        with open(result_obj, "r") as f:
            return json.load(f)["results"]

    # Otherwise, it is a dictionary. Process recursively
    else:
        return {key: load_results_data(value) for key, value in result_obj.items()}


def extract_data_category(data, category, names, noise_level):
    extracted_data = [
        [pair[category] for pair in data[name][noise_level] if category in pair.keys()]
        for name in names
    ]

    # Simply remove NaN's
    return extracted_data


def extract_memory_usage(data, names, noise_level):
    # Handle memory usage separately
    # GPU mem for ML, CPU mem for others
    extracted_data = [
        [
            pair["mem_gpu"] if "mem_gpu" in pair.keys() else pair["mem_cpu"]
            for pair in data[name][noise_level]
        ]
        for name in names
    ]
    return extracted_data


def colorize_boxplot(boxplot, colors):
    for idx in range(len(boxplot["boxes"])):
        color = colors[idx % len(colors)]
        plt.setp(boxplot["boxes"][idx], color=color)
        plt.setp(boxplot["medians"][idx], color="k")


def plot_category(data, subplot, colors, groups, ylabel, hline=None, lims=None):
    # Number of individual algorithms to be displayed
    num_units = len(data) // len(groups)
    positions = []
    label_positions = []
    vline_locations = []
    gap = 1

    for i in range(len(groups)):
        group_positions = [i * (num_units + gap) + j + 1 for j in range(num_units)]
        positions += group_positions
        label_positions += [(group_positions[0] + group_positions[-1]) / 2]

        if i != len(groups) - 1:
            vline_locations.append(positions[-1] + gap)

    # Plot itself
    boxplot = subplot.boxplot(
        data,
        positions=positions,
        patch_artist=True,
        flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "k"},
        widths=0.8,
    )

    # Apply colors and formats
    colorize_boxplot(boxplot, colors)

    # Vertical lines to separate groups
    for vline_location in vline_locations:
        subplot.axvline(vline_location, linestyle="--", color="k", linewidth=1)

    # Threshold, if given
    if hline is not None:
        subplot.axhline(hline, linestyle="--", color="r", linewidth=1)

    # Mark groups with ticks
    subplot.set_xticks(label_positions, [f"σ = {elem}" for elem in groups])

    # Other modifications
    if lims is not None:
        subplot.set_ylim(bottom=lims[0], top=lims[1])

    subplot.set_title(ylabel)
    subplot.grid(axis="y", which="both")
    subplot.set_yscale("log")
    subplot.minorticks_on()

    return boxplot


def generate_plot(results_data, names, colors):
    """Generate the box plots for a particular grid size
    results_data must be a dict one level after the data category
    (e.g. g0.05 within InteriorNet)
    """

    categories = ["error_position", "error_angle", "execution_time", "memory_usage"]
    titles = [
        "Translation Error (m)",
        "Rotation Error (rad)",
        "Execution Time (s)",
        "Memory use (Bytes)",
    ]

    # Collect all noises side-by-side
    plot_data = {category: [] for category in categories}
    noise_levels = ["0.00", "0.02", "0.05"]
    # noise_levels = ["0.00"]

    for noise_level in noise_levels:
        plot_data["error_position"] += extract_data_category(
            results_data, "error_position", names, noise_level
        )
        plot_data["error_angle"] += extract_data_category(
            results_data, "error_angle", names, noise_level
        )
        plot_data["execution_time"] += extract_data_category(
            results_data, "t_total", names, noise_level
        )
        plot_data["memory_usage"] += extract_memory_usage(
            results_data, names, noise_level
        )

    # Figure
    thresholds = [0.05 * 5, 0.174533, 60, 8e9]
    limits = [(1e-3, 4e1), (9e-6, 4e0), None, None]

    # plot_dict = {"subplots": [figure.add_subplot(1, 4, i + 1) for i in range(4)]}
    # plot_dict["boxplots"] = [
    #     plot_category(
    #         plot_data[categories[i]],
    #         plot_dict["subplots"][i],
    #         colors,
    #         noise_levels,
    #         titles[i],
    #         limits[i],
    #     )
    #     for i in range(len(categories))
    # ]

    plot_dict = {
        "figures": [plt.figure(figsize=[9, 4], tight_layout=True) for i in range(4)]
    }

    plot_dict["subplots"] = [
        plot_dict["figures"][i].add_subplot(1, 1, 1) for i in range(4)
    ]

    plot_dict["boxplots"] = [
        plot_category(
            plot_data[categories[i]],
            plot_dict["subplots"][i],
            colors,
            noise_levels,
            titles[i],
            thresholds[i],
            limits[i],
        )
        for i in range(4)
    ]

    # Shrink plots to make space
    padding = 0.1
    for subplot in plot_dict["subplots"]:
        box = subplot.get_position()
        subplot.set_position(
            pos=[box.x0, box.y0 - padding, box.width, box.height + padding]
        )

    # Legend
    # figure.legend(
    #     handles=plot_dict["boxplots"][0]["boxes"][: len(names)],
    #     labels=names,
    #     ncol=9,
    #     loc="lower center",
    #     mode="expand",
    # )
    for i, figure in enumerate(plot_dict["figures"]):
        figure.legend(
            handles=plot_dict["boxplots"][0]["boxes"][: len(names)],
            labels=names,
            ncol=5,
            loc="lower center",
            mode="expand",
        )
        figure.savefig(f"figures/{categories[i]}.pdf", format="pdf")

    # Display
    # Keeps reverting settings after show. Display rendered image instead
    # plt.show()

    # plt.show()


def main():
    # Colors are independent. Order dictated by the list below
    # Must match the name in the data
    names = [
        "Consensus",
        "Tomographic-TEASER++",
        "FPFH-RANSAC",
        "FPFH-TEASER++",
        "ORB-TEASER++",
        "DeepGlobalRegistration",
        "GeoTransformer",
        "RoITr",
        "BUFFER",
    ]

    # Colors from http://vrl.cs.brown.edu/color
    colors = [
        "#e51d1d",
        "#377eb8",
        "#4eaf49",
        "#974da2",
        "#ff8000",
        "#096013",
        "#f82387",
        "#32a190",
        "#852405",
    ]
    # File correspondences
    root_folder = "results"

    # InteriorNet
    raw_data = {
        "InteriorNet": {
            "g0.05": {
                "Consensus": {
                    "0.00": os.path.join(
                        root_folder, "Consensus", "Consensus-2024-05-27-16-09-41.json"
                    ),
                    "0.02": os.path.join(
                        root_folder, "Consensus", "Consensus-2024-05-27-16-10-02.json"
                    ),
                    "0.05": os.path.join(
                        root_folder, "Consensus", "Consensus-2024-05-27-16-11-34.json"
                    ),
                },
                "FPFH-RANSAC": {
                    "0.00": os.path.join(
                        root_folder, "Consensus", "FPFH-RANSAC-2024-05-27-16-13-44.json"
                    ),
                    "0.02": os.path.join(
                        root_folder, "Consensus", "FPFH-RANSAC-2024-05-27-16-14-19.json"
                    ),
                    "0.05": os.path.join(
                        root_folder, "Consensus", "FPFH-RANSAC-2024-05-27-16-26-34.json"
                    ),
                },
                "FPFH-TEASER++": {
                    "0.00": os.path.join(
                        root_folder, "Consensus", "FPFH-TEASER-2024-05-27-16-50-55.json"
                    ),
                    "0.02": os.path.join(
                        root_folder, "Consensus", "FPFH-TEASER-2024-05-27-16-51-30.json"
                    ),
                    "0.05": os.path.join(
                        root_folder, "Consensus", "FPFH-TEASER-2024-05-27-17-03-44.json"
                    ),
                },
                "Tomographic-TEASER++": {
                    "0.00": os.path.join(
                        root_folder, "Consensus", "ORB-TEASER-2024-05-27-17-28-04.json"
                    ),
                    "0.02": os.path.join(
                        root_folder, "Consensus", "ORB-TEASER-2024-05-27-17-28-54.json"
                    ),
                    "0.05": os.path.join(
                        root_folder, "Consensus", "ORB-TEASER-2024-05-27-17-31-17.json"
                    ),
                },
                "ORB-TEASER++": {
                    "0.00": os.path.join(
                        root_folder, "Consensus", "ORB-TEASER-2024-05-27-17-34-04.json"
                    ),
                    "0.02": os.path.join(
                        root_folder, "Consensus", "ORB-TEASER-2024-05-27-17-34-54.json"
                    ),
                    "0.05": os.path.join(
                        root_folder, "Consensus", "ORB-TEASER-2024-05-27-17-36-36.json"
                    ),
                },
                "BUFFER": {
                    "0.00": os.path.join(
                        root_folder,
                        "BUFFER",
                        "buffer_interiornet_gsize0-05-noise0-00_2024-05-26-19-00-24.json",
                    ),
                    "0.02": os.path.join(
                        root_folder,
                        "BUFFER",
                        "buffer_interiornet_gsize0-05-noise0-02_2024-05-26-19-01-22.json",
                    ),
                    "0.05": os.path.join(
                        root_folder,
                        "BUFFER",
                        "buffer_interiornet_gsize0-05-noise0-05_2024-05-26-19-03-47.json",
                    ),
                },
                "DeepGlobalRegistration": {
                    "0.00": os.path.join(
                        root_folder,
                        "DeepGlobalRegistration",
                        "dgr_interiornet_gsize0-05-noise0-00_2024-05-26-19-00-31.json",
                    ),
                    "0.02": os.path.join(
                        root_folder,
                        "DeepGlobalRegistration",
                        "dgr_interiornet_gsize0-05-noise0-02_2024-05-26-19-02-18.json",
                    ),
                    "0.05": os.path.join(
                        root_folder,
                        "DeepGlobalRegistration",
                        "dgr_interiornet_gsize0-05-noise0-05_2024-05-26-19-30-36.json",
                    ),
                },
                "GeoTransformer": {
                    "0.00": os.path.join(
                        root_folder,
                        "GeoTransformer",
                        "GeoTransformer_interiornet_gsize0-05-noise0-00_2024-05-26-19-10-53.json",
                    ),
                    "0.02": os.path.join(
                        root_folder,
                        "GeoTransformer",
                        "GeoTransformer_interiornet_gsize0-05-noise0-02_2024-05-26-19-15-43.json",
                    ),
                    "0.05": os.path.join(
                        root_folder,
                        "GeoTransformer",
                        "GeoTransformer_interiornet_gsize0-05-noise0-05_2024-05-26-19-35-40.json",
                    ),
                },
                "RoITr": {
                    "0.00": os.path.join(
                        root_folder,
                        "RoITr",
                        "roitr_interiornet_gsize0-05-noise0-00_2024-05-26-19-07-09.json",
                    ),
                    "0.02": os.path.join(
                        root_folder,
                        "RoITr",
                        "roitr_interiornet_gsize0-05-noise0-02_2024-05-26-19-08-02.json",
                    ),
                    "0.05": os.path.join(
                        root_folder,
                        "RoITr",
                        "roitr_interiornet_gsize0-05-noise0-05_2024-05-26-19-09-01.json",
                    ),
                },
            }
        }
    }

    # Convert from files to actual data
    results_data = load_results_data(raw_data)

    test = generate_plot(results_data["InteriorNet"]["g0.05"], names, colors)


if __name__ == "__main__":
    main()
