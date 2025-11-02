import argparse
from train import train_policy
from inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Probabilistic Decision Framework (PDF)")
    parser.add_argument("--mode", type=str, default="inference", choices=["train", "inference"], help="Run mode")
    parser.add_argument("--dataset", type=str, default="mmlu", help="Dataset name")
    args = parser.parse_args()

    if args.mode == "train":
        train_policy()
    else:
        run_inference(args.dataset)

if __name__ == "__main__":
    main()
