import random
from functools import partial
from copy import copy

from argparse import ArgumentParser

import torch

from model import EsmcGoTermClassifier

from torch.cuda import is_available as cuda_is_available

import obonet

import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("qt5agg")


def main():
    parser = ArgumentParser(
        description="Predict the rank of the gene ontology (GO) terms associated with a protein sequence."
    )

    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--go_db_path", default="./dataset/go-basic.obo", type=str)
    parser.add_argument("--context_length", default=1026, type=int)
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

    graph = obonet.read_obo(args.go_db_path)

    assert nx.is_directed_acyclic_graph(graph), "Invalid GO graph, use basic DAG."

    print("Gene Ontology graph loaded successfully.")

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )

    model_args = checkpoint["model_args"]

    model = EsmcGoTermClassifier.from_esm_pretrained(**model_args)

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

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

        with torch.no_grad():
            y_pred = model.forward(input_ids)

        probabilities = torch.sigmoid(y_pred.squeeze(0))

        go_term_probabilities = {
            model.id2label[index]: probability.item()
            for index, probability in enumerate(probabilities)
            if probability > args.top_p
        }

        # Fix up the predictions by leveraging the GO DAG hierarchy.
        for go_term, parent_probability in copy(go_term_probabilities).items():
            for descendant in nx.descendants(graph, go_term):
                if descendant in go_term_probabilities:
                    child_probability = go_term_probabilities[descendant]
                else:
                    child_probability = 0.0

                go_term_probabilities[descendant] = max(
                    parent_probability,
                    child_probability,
                )

        subgraph = graph.subgraph(go_term_probabilities.keys())

        color_intensities = [
            go_term_probabilities[go_term] for go_term in subgraph.nodes()
        ]

        node_labels = {
            go_term: f"{go_term}\n{data["name"]}"
            for go_term, data in subgraph.nodes(data=True)
        }

        plt.figure(figsize=(10, 8))
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
