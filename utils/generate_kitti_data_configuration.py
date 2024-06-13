import itertools
import json
import argparse
import os

# Need these for all options
# Only the sequences below revisit places
sequence_ids = ["00", "02", "05", "06", "07"]
trajectory_pairs = [(1, 2), (2, 1)]

# Product of these will be different configuration files
grid_sizes = ["1.50", "1.00", "0.50", "0.20"]
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
    os.makedirs(args.output)

    for outer_prod in itertools.product(grid_sizes, noise_levels):
        pairs = []

        grid_size = outer_prod[0]
        noise_level = outer_prod[1]

        for inner_prod in itertools.product(sequence_ids, trajectory_pairs):
            seq_id = inner_prod[0]
            trajectory_pair = inner_prod[1]

            pair = [
                f"kitti{seq_id}-g{grid_size}-traj{trajectory_pair[0]}-n{noise_level}.pcd",
                f"kitti{seq_id}-g{grid_size}-traj{trajectory_pair[1]}-n{noise_level}.pcd",
            ]
            pairs.append(pair)

        config_dict = {"root": args.root, "pairs": pairs}
        file_name = f"gsize{grid_size}-noise{noise_level}".replace(".", "-") + ".json"
        file_path = os.path.join(args.output, file_name)

        print("Full config:")
        print(json.dumps(config_dict, indent=2))

        with open(file_path, "w") as f:
            json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    main()
