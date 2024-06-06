import os
import subprocess

# Pairing for parameter and data configuration
studies = {
    "interiornet": {
        "parameter_configs": [
            "consensus-interiornet.json",
            "fpfh-ransac-interiornet.json",
            "fpfh-teaser-interiornet.json",
            "orb-teaser-2d-interiornet.json",
            "orb-teaser-3d-interiornet.json",
        ],
        "data_configs": [
            "gsize0-05-noise0-00.json",
            "gsize0-05-noise0-02.json",
            "gsize0-05-noise0-05.json",
        ],
    },
    "kitti": {
        "parameter_configs": [
            "consensus-kitti.json",
            "fpfh-ransac-kitti.json",
            "fpfh-teaser-kitti.json",
            "orb-teaser-2d-kitti.json",
            "orb-teaser-3d-kitti.json",
        ],
        "data_configs": [
            "gsize0-05-noise0-00.json",
        ],
    },
}

PKG = "Consensus"

# If DATA_DIR is not set, we should fail
DATA_DIR = os.environ.get("DATA_DIR")
if DATA_DIR is None:
    print("Environment variable DATA_DIR not set")
    exit(-1)

command_base = " ".join(
    [
        f"singularity run --nv --no-home",
        f"--bind {PKG}/workspace:/workspace",
        f"--bind results/{PKG}:/results",
        f"--bind data-config:/data-config",
        f"--bind {DATA_DIR}:/data",
        f"{PKG}/{PKG}.sif",
    ]
)


def main():
    # Ensure folders exist
    os.makedirs(f"results/{PKG}", exist_ok=True)

    for name, study in studies.items():
        print(f">> Running {name}")

        for parameter_config in study["parameter_configs"]:
            for data_config in study["data_configs"]:
                print(f"> Parameter: {parameter_config} Data: {data_config}")

                command = " ".join(
                    [
                        command_base,
                        f"--parameter_config config/{parameter_config}",
                        f"--data_config /data-config/{name}/{data_config}",
                    ]
                )
                print(command)
                subprocess.run(["/bin/bash", "-c", command])
                print()


if __name__ == "__main__":
    main()
