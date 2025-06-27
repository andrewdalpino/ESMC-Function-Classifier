# ESMC Protein Function Predictor

An Evolutionary-scale Model (ESM) for protein function prediction from amino acid sequences using the Gene Ontology (GO). Based on the ESM Cambrian Transformer architecture, pre-trained on [UniRef](https://www.uniprot.org/help/uniref), [MGnify](https://www.ebi.ac.uk/metagenomics), and the Joint Genome Institute's database and fine-tuned on the [AmiGO](https://huggingface.co/datasets/andrewdalpino/AmiGO) protein function dataset, this model predicts the GO subgraph for a particular protein sequence - giving you insight into the molecular function, biological process, and location of the activity inside the cell.

## What are GO terms?

> "The Gene Ontology (GO) is a concept hierarchy that describes the biological function of genes and gene products at different levels of abstraction (Ashburner et al., 2000). It is a good model to describe the multi-faceted nature of protein function."

> "GO is a directed acyclic graph. The nodes in this graph are functional descriptors (terms or classes) connected by relational ties between them (is_a, part_of, etc.). For example, terms 'protein binding activity' and 'binding activity' are related by an is_a relationship; however, the edge in the graph is often reversed to point from binding towards protein binding. This graph contains three subgraphs (subontologies): Molecular Function (MF), Biological Process (BP), and Cellular Component (CC), defined by their root nodes. Biologically, each subgraph represent a different aspect of the protein's function: what it does on a molecular level (MF), which biological processes it participates in (BP) and where in the cell it is located (CC)."

From [CAFA 5 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data)


## Cloning the Repo

You'll need the code in the repository to run the model. To clone the repo onto your local machine enter the command like in the example below.

```sh
git clone https://github.com/andrewdalpino/ESMC-Function-Classifier
```

## Install Project Dependencies

Project dependencies are specified in the `requirements.txt` file. You can install them with [pip](https://pip.pypa.io/en/stable/) using the following command from the project root. We recommend using a virtual environment such as `venv` to keep package dependencies on your system tidy.

```
python -m venv ./.venv

source ./.venv/bin/activate

pip install -r requirements.txt
```

### Using a Pretrained Model

Since the HuggingFace [Transformers](https://github.com/huggingface/transformers) library supports the [ESM](https://huggingface.co/docs/transformers/en/model_doc/esm) architecture natively, we can start protein function calling quickly in just a few lines of code.

```python
from model import EsmcGoTermClassifier

model_name = "andrewdalpino/ESMC-300M-Protein-Function"

model = EsmcGoTermClassifier.from_pretrained(model_name)
```

## References:

>- M. Ashburner, et al. Gene Ontology: tool for the unification of biology, 2000.