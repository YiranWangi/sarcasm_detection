"""Train MT5 to generate Chinese labels, then evaluate the held-out test split."""

from training import run_training


if __name__ == "__main__":
    run_training("t5")
