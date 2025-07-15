import random
from functools import partial
from collections import defaultdict
from copy import copy

from argparse import ArgumentParser

import torch

from src.esmc_function_classifier.model import EsmcGoTermClassifier

from torch.cuda import is_available as cuda_is_available

import obonet

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")


def main():
    parser = ArgumentParser(
        description="Predict the gene ontology (GO) subgraph associated with a protein sequence."
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--go_db_path", default="./dataset/go-basic.obo", type=str)
    parser.add_argument("--context_length", default=2048, type=int)
    parser.add_argument("--top_p", default=0.5, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.context_length < 1:
        raise ValueError(
            f"Context length must be greater than 0, {args.context_length} given."
        )

    if args.top_p < 0.0 or args.top_p > 1.0:
        raise ValueError(f"Top p must be between 0 and 1, {args.top_p} given.")

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    torch.set_float32_matmul_precision("high")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )

    model_args = checkpoint["model_args"]

    model = EsmcGoTermClassifier.from_esm_pretrained(**model_args)

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    model.merge_lora_parameters()

    graph = obonet.read_obo(args.go_db_path)

    model.load_gene_ontology(graph)

    model.eval()

    print("Checkpoint loaded successfully.")

    plot_subgraph = partial(
        nx.draw_networkx,
        node_size=2000,
        font_size=9,
        cmap="PiYG",
        vmin=0,
        vmax=1,
        with_labels=True,
        arrowsize=20,
    )

    while True:
        sequence = input("Enter a sequence: ").replace(" ", "").replace("\n", "")

        out = model.tokenizer(
            sequence,
            max_length=args.context_length,
            truncation=True,
        )

        input_ids = out["input_ids"]

        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(args.device)
        )

        subgraph, go_term_probabilities = model.predict_subgraph(
            input_ids, top_p=args.top_p
        )

        color_intensities = [
            go_term_probabilities[go_term] for go_term in subgraph.nodes()
        ]

        node_labels = {
            go_term: f"{go_term}\n{data["name"]}"
            for go_term, data in subgraph.nodes(data=True)
        }

        plt.figure(figsize=(12, 10))
        plt.title("Gene Ontology Subgraphs")

        plot_subgraph(
            subgraph,
            node_color=color_intensities,
            labels=node_labels,
        )

        plt.show()

        if "y" not in input("Go again? (yes|no): ").lower():
            break


if __name__ == "__main__":
    main()
