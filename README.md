# GenDLStudy

Study notes of
[Generative Deep Learning, 2nd Edition](https://www.oreilly.com/library/view/generative-deep-learning/9781098134174/)
by David Foster.

## Symposia

* 2024.6.1: [Preface](Preface.md)
* 2024.6.8: [Chapter 1](Chapter1.md)
* 2024.6.15: [Chapter 2](Chapter2.md)
* 2024.6.22: part 1 of [Chapter 3](Chapter3.md)

## Running Codes

Some Python scripts are converted from Jupyter notebook to avoid
Base64 strings messed up version history.
These scripts can be executed in 3 ways:

First as Python script:
```sh
cd scripts
poetry run ipython <script>.py
```

Or convert to Jupyter notebook and play with it interactively:
```sh
poetry jupytext --to notebook <script.py>
```

Lastly you can convert and execute it in one go:
```sh
poetry jupytext --to notebook --execute <script.py>
```

Choose one from them according to your needs.
