import json
import argparse
import os

# Need these for all options
# Only the sequences below revisit places
sequence_ids = ["1"]
trajectory_pairs = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

# Product of these will be different configuration files
grid_sizes = ["0.10", "0.05"]
noise_levels = ["0.00"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="data-config",
        help="Folder name to generate the configurations",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data",
        help="Root directory for the data",
    )

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    for grid_size in grid_sizes:
        pairs = []

        for trajectory_pair in trajectory_pairs:
            pair = [
                f"a1-g{grid_size}-traj{trajectory_pair[0]}-n0.00.pcd",
                f"a1-g{grid_size}-traj{trajectory_pair[1]}-n0.00.pcd",
            ]
            pairs.append(pair)

        config_dict = {"root": args.root, "pairs": pairs}
        file_name = f"gsize{grid_size}-noise0-00".replace(".", "-") + ".json"
        file_path = os.path.join(args.output, file_name)

        print("Full config:")
        print(json.dumps(config_dict, indent=2))

        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    main()
