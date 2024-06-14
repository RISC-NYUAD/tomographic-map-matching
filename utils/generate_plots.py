import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import math
import itertools


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


def plot_category(
    data, subplot, colors, groups, ylabel=None, title=None, hline=None, lims=None
):
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
    if len(groups) > 1:
        subplot.set_xticks(label_positions, [f"σ = {elem}" for elem in groups])
    else:
        subplot.set_xticks([])

    # Other modifications
    if lims is not None:
        subplot.set_ylim(bottom=lims[0], top=lims[1])

    if ylabel is not None:
        subplot.set_ylabel(ylabel)

    if title is not None:
        subplot.set_title(title, fontdict={"fontsize": 11})

    subplot.grid(axis="y", which="both")
    subplot.set_yscale("log")
    subplot.minorticks_on()

    return boxplot


def generate_split_plots(
    results_data, names, colors, thresholds, limits, noise_levels, title
):
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
    tight_layout_params = {"pad": 1.00, "rect": (0, 0, 0.75, 1)}
    plot_dict = {
        "figures": [
            plt.figure(figsize=[9, 3], tight_layout=tight_layout_params)
            for i in range(4)
        ]
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
            title=titles[i],
            hline=thresholds[i],
            lims=limits[i],
        )
        for i in range(4)
    ]

    for i, figure in enumerate(plot_dict["figures"]):
        figure.legend(
            handles=plot_dict["boxplots"][0]["boxes"][: len(names)],
            labels=names,
            labelspacing=1.01,
            loc="right",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )
        figure.savefig(
            f"figures/{title}-{categories[i]}.pdf", format="pdf", bbox_inches="tight"
        )


def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])


def generate_combined_plot(
    results_data, names, colors, thresholds, limits, noise_levels, title
):
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
    tight_layout_params = {"pad": 1.00, "w_pad": 0.2, "rect": (0, 0.11, 1, 1)}

    plot_dict = {
        "figures": [plt.figure(figsize=[9, 3], tight_layout=tight_layout_params)]
    }

    plot_dict["subplots"] = [
        plot_dict["figures"][0].add_subplot(1, 4, i + 1) for i in range(4)
    ]

    plot_dict["boxplots"] = [
        plot_category(
            plot_data[categories[i]],
            plot_dict["subplots"][i],
            colors,
            noise_levels,
            title=titles[i],
            hline=thresholds[i],
            lims=limits[i],
        )
        for i in range(4)
    ]
    figure = plot_dict["figures"][0]

    ncol = 4 if len(names) > 6 else 3
    handles = flip(plot_dict["boxplots"][0]["boxes"][: len(names)], ncol)
    labels = flip(names, ncol)

    figure.legend(
        handles=handles,
        labels=labels,
        ncol=ncol,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        # columnspacing=0.1,
        labelspacing=0.5,
        frameon=False,
        mode={"expand": True},
    )
    figure.savefig(f"figures/{title}.pdf", format="pdf", bbox_inches="tight")


def main():
    # Colors are independent. Order dictated by the list below
    # Must match the name in the data
    names = [
        "Consensus",
        "Tomographic-TEASER++",
        "ORB-TEASER++",
        "FPFH-RANSAC",
        "FPFH-TEASER++",
        "DeepGlobalRegistration",
        "BUFFER",
        "GeoTransformer",
        "RoITr",
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

    # fmt: off
    raw_data = {
        "InteriorNet": {
            "g0.05": {
                "Consensus": {
                    "0.00": os.path.join(root_folder, "Consensus", "Consensus-2024-06-06-14-24-06.json"),
                    "0.02": os.path.join(root_folder, "Consensus", "Consensus-2024-06-06-14-24-37.json"),
                    "0.05": os.path.join(root_folder, "Consensus", "Consensus-2024-06-06-14-27-23.json"),
                },
                "FPFH-RANSAC": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-RANSAC-2024-06-06-14-31-35.json"),
                    "0.02": os.path.join(root_folder, "Consensus", "FPFH-RANSAC-2024-06-06-14-32-35.json"),
                    "0.05": os.path.join(root_folder, "Consensus", "FPFH-RANSAC-2024-06-06-15-01-55.json"),
                },
                "FPFH-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-TEASER-2024-06-06-16-03-39.json"),
                    "0.02": os.path.join(root_folder, "Consensus", "FPFH-TEASER-2024-06-06-16-04-51.json"),
                    "0.05": os.path.join(root_folder, "Consensus", "FPFH-TEASER-2024-06-06-16-36-00.json"),
                },
                "Tomographic-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-06-17-38-05.json"),
                    "0.02": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-06-17-39-12.json"),
                    "0.05": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-06-17-47-20.json"),
                },
                "ORB-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-06-18-01-16.json"),
                    "0.02": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-06-18-04-02.json"),
                    "0.05": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-06-18-15-53.json"),
                },
                "BUFFER": {
                    "0.00": os.path.join(root_folder, "BUFFER", "buffer_interiornet_gsize0-05-noise0-00_2024-06-06-14-39-07.json"),
                    "0.02": os.path.join(root_folder, "BUFFER", "buffer_interiornet_gsize0-05-noise0-02_2024-06-06-14-41-02.json"),
                    "0.05": os.path.join(root_folder, "BUFFER", "buffer_interiornet_gsize0-05-noise0-05_2024-06-06-14-43-28.json"),
                },
                "DeepGlobalRegistration": {
                    "0.00": os.path.join(root_folder, "DeepGlobalRegistration", "dgr_interiornet_gsize0-05-noise0-00_2024-06-06-14-46-06.json"),
                    "0.02": os.path.join(root_folder, "DeepGlobalRegistration", "dgr_interiornet_gsize0-05-noise0-02_2024-06-06-14-48-42.json"),
                    "0.05": os.path.join(root_folder, "DeepGlobalRegistration", "dgr_interiornet_gsize0-05-noise0-05_2024-06-06-15-15-48.json"),
                },
                "GeoTransformer": {
                    "0.00": os.path.join(root_folder, "GeoTransformer", "GeoTransformer_interiornet_gsize0-05-noise0-00_2024-06-06-14-15-37.json"),
                    "0.02": os.path.join(root_folder, "GeoTransformer", "GeoTransformer_interiornet_gsize0-05-noise0-02_2024-06-06-14-17-22.json"),
                    "0.05": os.path.join(root_folder, "GeoTransformer", "GeoTransformer_interiornet_gsize0-05-noise0-05_2024-06-06-14-34-34.json"),
                },
                "RoITr": {
                    "0.00": os.path.join(root_folder, "RoITr", "roitr_interiornet_gsize0-05-noise0-00_2024-06-06-14-55-32.json"),
                    "0.02": os.path.join(root_folder, "RoITr", "roitr_interiornet_gsize0-05-noise0-02_2024-06-06-14-57-13.json"),
                    "0.05": os.path.join(root_folder, "RoITr", "roitr_interiornet_gsize0-05-noise0-05_2024-06-06-14-58-12.json"),
                },
            }
        },
        "KITTI": {
            "g0.20": {
                "Consensus": {
                    "0.00": os.path.join(root_folder, "Consensus", "Consensus-2024-06-07-16-45-58.json"),
                },
                "FPFH-RANSAC": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-RANSAC-2024-06-07-17-03-54.json"),
                },
                "Tomographic-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-07-17-38-48.json"),
                },
                "ORB-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-07-18-05-30.json"),
                },
                "BUFFER": {
                    "0.00": os.path.join(root_folder, "BUFFER", "buffer_kitti_gsize0-20-noise0-00_2024-06-06-14-48-43.json"),
                },
                "DeepGlobalRegistration": {
                    "0.00": os.path.join(root_folder, "DeepGlobalRegistration", "dgr_kitti_gsize0-20-noise0-00_2024-06-06-16-20-25.json"),
                },
                "GeoTransformer": {
                    "0.00": os.path.join(root_folder, "GeoTransformer", "GeoTransformer_kitti_gsize0-20-noise0-00_2024-06-06-15-07-05.json"),
                },
            },
            "g0.50": {
                "Consensus": {
                    "0.00": os.path.join(root_folder, "Consensus", "Consensus-2024-06-07-16-42-38.json"),
                },
                "FPFH-RANSAC": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-RANSAC-2024-06-07-16-59-52.json"),
                },
                "FPFH-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-TEASER-2024-06-07-17-26-14.json"),
                },
                "Tomographic-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-07-17-38-09.json"),
                },
                "ORB-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-07-18-02-28.json"),
                },
                "BUFFER": {
                    "0.00": os.path.join(root_folder, "BUFFER", "buffer_kitti_gsize0-50-noise0-00_2024-06-06-14-46-47.json"),
                },
                "DeepGlobalRegistration": {
                    "0.00": os.path.join(root_folder, "DeepGlobalRegistration", "dgr_kitti_gsize0-50-noise0-00_2024-06-06-16-18-41.json"),
                },
                "GeoTransformer": {
                    "0.00": os.path.join(root_folder, "GeoTransformer", "GeoTransformer_kitti_gsize0-50-noise0-00_2024-06-06-15-04-25.json"),
                },
            }
        },
        "A1": {
            "g0.05": {
                "Consensus": {
                    "0.00": os.path.join(root_folder, "Consensus", "Consensus-2024-06-14-10-59-22.json"),
                },
                "FPFH-RANSAC": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-RANSAC-2024-06-14-11-04-18.json"),
                },
                "FPFH-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "FPFH-TEASER-2024-06-14-11-17-17.json"),
                },
                "Tomographic-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-14-11-31-15.json"),
                },
                "ORB-TEASER++": {
                    "0.00": os.path.join(root_folder, "Consensus", "ORB-TEASER-2024-06-14-11-43-03.json"),
                },
                "BUFFER": {
                    "0.00": os.path.join(root_folder, "BUFFER", "buffer_a1-vdbmap_gsize0-05-noise0-00_2024-06-14-10-58-19.json"),
                },
                "DeepGlobalRegistration": {
                    "0.00": os.path.join(root_folder, "DeepGlobalRegistration", "dgr_a1-vdbmap_gsize0-05-noise0-00_2024-06-14-10-56-47.json"),
                },
                "RoITr": {
                    "0.00": os.path.join(root_folder, "RoITr", "roitr_a1-vdbmap_gsize0-05-noise0-00_2024-06-14-10-58-46.json"),
                },
            }
        }
    }
    # fmt: on

    # Convert from files to actual data
    results_data = load_results_data(raw_data)

    # InteriorNet
    thresholds = [0.05 * 5, 0.174533, 60, 1.6e10]
    limits = [(5e-4, 2e1), None, None, None]
    noise_levels = ["0.00", "0.02", "0.05"]
    generate_split_plots(
        results_data["InteriorNet"]["g0.05"],
        names,
        colors,
        thresholds,
        limits,
        noise_levels,
        "interiornet05",
    )

    # KITTI 0.5
    thresholds = [0.50 * 5, 0.174533, 60, 1.6e10]
    limits = [None, None, None, None]
    noise_levels = ["0.00"]
    generate_combined_plot(
        results_data["KITTI"]["g0.50"],
        names[:-2],
        colors[:-2],
        thresholds,
        limits,
        noise_levels,
        "kitti50",
    )

    # KITTI 0.2
    thresholds = [0.20 * 5, 0.174533, 60, 1.6e10]
    limits = [None, None, None, None]
    noise_levels = ["0.00"]
    generate_combined_plot(
        results_data["KITTI"]["g0.20"],
        names[:4] + names[5:-2],
        colors[:4] + colors[5:-2],
        thresholds,
        limits,
        noise_levels,
        "kitti20",
    )

    # Experimental
    thresholds = [0.05 * 5, 0.174533, 60, 1.6e10]
    limits = [None, None, None, None]
    noise_levels = ["0.00"]
    generate_combined_plot(
        results_data["A1"]["g0.05"],
        names[:-2],
        colors[:-2],
        thresholds,
        limits,
        noise_levels,
        "a1",
    )

    plt.show()


if __name__ == "__main__":
    main()
