import argparse

from src import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()

    # Builds current and spike tensors with DT=0.1ms
    # Note: You might want to set manifest_file if you have already downloaded the data from the Allen Institute
    builder = datasets.NoiseBuilder()
    builder.build(f"{args.root}/Blocks/data/ephys/train", noise_type="noise1")
    builder.build(f"{args.root}/Blocks/data/ephys/test", noise_type="noise2")

    # Builds current tensors with DT=0.05ms
    builder = datasets.NoiseBuilder()
    builder.build(f"{args.root}/Blocks/data/ephys/train", noise_type="noise1", target_sampling_rate=20000)
    builder.build(f"{args.root}/Blocks/data/ephys/test", noise_type="noise2", target_sampling_rate=20000)


if __name__ == "__main__":
    main()
