# Seznam task

To install dependencies one can use [poetry](https://python-poetry.org/):
- install poetry: [link](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions)
- to install dependencies simply run (being in the repo root):
`poetry install`

1. **Methods used/tried**:
- Naive-Bayes
- LSTM
- LSTM with embeddings

2. **Sources**:
   - main ntb:                     _sez_eda.ipynb_
   - running NN in Colab:          _lstm.ipynb_ (also on [Colab](https://colab.research.google.com/drive/1ugKZxhmy71eURJSDRoJaJhG65jPUIhNc)
   - funcs, utils here:            _dl.py, utils.py, stop_words_cz.py_

3. **Resulted** in model with the following metrics:
   f1-score 0.7-0.99 for different classes.
   Of course, there is place for improvement.