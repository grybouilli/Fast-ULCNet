#!/usr/bin/env python3

import re
import sys
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def parse_log(filename):
    epochs = []
    train_loss = []
    val_loss = []

    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|\s+"
        r"train_loss=([0-9.eE+-]+)\s+\|\s+"
        r"val_loss=([0-9.eE+-]+)"
    )

    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)

            if match:
                epoch = int(match.group(1))
                train = float(match.group(2))
                val = float(match.group(3))

                epochs.append(epoch)
                train_loss.append(train)
                val_loss.append(val)

    return epochs, train_loss, val_loss


def main():
    parser = ArgumentParser()
    parser.add_argument("log_file", type=str)
    args = parser.parse_args()

    filename = args.log_file

    epochs, train_loss, val_loss = parse_log(filename)

    if not epochs:
        print("No epoch data found in log.")
        sys.exit(1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_loss,
        label="Training loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        val_loss,
        label="Validation loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()