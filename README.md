# Hierarchical Compositionality for An Assistive AI Agent

Code, data and results for the paper [Hierarchical Compositionality for An
Assistive AI Agent](https://arxiv.org/abs/2608.10330).

![Motivating Example](docs/figures/fig_motivating.png)
![Framework](docs/figures/framework_2.png)
![Compositional Hierarchy](docs/figures/compositional_hierarchy.png)

## Installation

Python 3.8+, Java 11+ and clingo 5.7.1+ are required. Clone the repository and
install the dependencies:

```
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
conda install -c potassco clingo
```

The LLM baselines B1 and B2 call the OpenAI API:

```
export OPENAI_API_KEY="Replace with your OpenAI API key"
```

Tested on:

* macOS, Apple M4 Pro (CPU/GPU), 24 GB RAM
* Windows 11, Intel Core i9-14900KF, 48 GB RAM, NVIDIA RTX 4090
* Python 3.12.2, OpenJDK 23.0.1, clingo 5.7.1
* rdflib 7.1.1, nltk 3.9.1, spaCy 3.7.2, openai 1.47.0, numpy 1.26.4, pandas 2.2.3
* LLM baselines: `gpt-5.1`, temperature 0
* Random baseline B0: fixed seed 42

## Usage

To run one user at one ambiguity level (`--user user1` to `user5`, `--level a1`
to `a4`), run:

```
python run_all.py --user user1 --level a1
```

To run all users at all levels, run:

```
python run_all.py
```

Each run first derives the world states with SPARC. The outputs are written to
`result/`. Runs use the no-ask protocol of the paper; set
`BASELINE8_FORCE_NO_ASK=0` to let the methods ask for clarification instead.

To resolve a command interactively, run:

```
python main.py
```

When prompted, input an English command and then its ASP-format goal template,
where `__OBJ__` marks the object to resolve. For example:

* `has(user, __OBJ__)`
* `on(table, __OBJ__)`
* `inside(microwave_oven, __OBJ__)`
* `heated(__OBJ__)`
* `filled(__OBJ__, kettle)`
* `switched_on(__OBJ__)`

## Data Sources and Tools

* [NOVA](https://onlinelibrary.wiley.com/doi/pdf/10.1111/tops.70037)
* [WordNet](https://wordnet.princeton.edu/)
* [SPARC](https://github.com/iensen/sparc)
* [spaCy `en_core_web_sm`](https://spacy.io/models/en)

## Citation

```bibtex
@misc{fu2026hierarchicalcompositionalityassistiveai,
      title={Hierarchical Compositionality for An Assistive AI Agent},
      author={Tianyi Fu and Mohan Sridharan},
      year={2026},
      eprint={2608.10330},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.10330},
}
```

## License

* Code: MIT (`LICENSE`)
* Data and results: CC BY 4.0
