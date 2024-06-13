import itertools
import json
import argparse
import os

# Need these for all options
room_ids = ["4NDAO", "LITSO", "SULLS", "HSTAH", "AAKRW"]
trajectory_pairs = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]

# Product of these will be different configuration files
grid_sizes = ["0.02", "0.05", "0.10", "0.20"]
noise_levels = ["0.00", "0.02", "0.05"]


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

        for inner_prod in itertools.product(room_ids, trajectory_pairs):
            room_id = inner_prod[0]
            trajectory_pair = inner_prod[1]

            pair = [
                f"{room_id}-g{grid_size}-traj{trajectory_pair[0]}-n{noise_level}.pcd",
                f"{room_id}-g{grid_size}-traj{trajectory_pair[1]}-n{noise_level}.pcd",
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
