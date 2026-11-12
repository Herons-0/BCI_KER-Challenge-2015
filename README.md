# BCI Challenge @ NER 2015: error detection with an MLP

> **Status: unfinished.** This is an early attempt at the [Kaggle BCI Challenge](https://www.kaggle.com/competitions/inria-bci-challenge/overview). `model.py` extracts the post-feedback EEG windows and defines an MLP, but the model has not been trained or evaluated yet, so there are no results. See [Known issues](#known-issues) for what is left.

## Problem

The "P300 speller" is a well-known brain-computer interface (BCI) paradigm. It uses electroencephalography (EEG) and the P300 response, which is evoked by rare, attended stimuli, to select items on a screen. In this experiment each subject spelled words from a 6 × 6 matrix of letters and numbers. Items flash in groups in random order, and the speller selects the item whose flashes produce the most target-like response.

After each selection the subject sees feedback. The challenge is to detect, from the EEG recorded after that feedback, when the selected item was wrong, so that the speller could correct its own errors automatically (the idea studied in reference 1).

## Approach

The plan is to train a plain multilayer perceptron (MLP) on the raw post-feedback window, instead of the hand-engineered features with linear models and ensembles used by most challenge entries (see the winning solution, reference 2). Compact CNNs such as EEGNet (reference 3) have since been evaluated on this same dataset, so the open question is how close a simple MLP gets.

### Input features (per feedback event)

1. **EEG**, from 300 ms after feedback (roughly when the brain registers it) to 1300 ms after (when the feedback leaves the screen): 56 channels × 200 samples at 200 Hz = 11,200 values.
2. **EOG** (eye movement) over the same window: 200 values.
3. **Meta features**: time of the feedback event, session number, word number, and character position within the word.

### Model

scikit-learn `MLPClassifier` with ReLU, Adam, and early stopping on a 20% validation split. The full settings are in `model.py`.

## Usage

Download the challenge data from Kaggle, then point the script at the folder of training CSVs:

```bash
pip install numpy pandas scikit-learn
python model.py path/to/train
```

## Known issues

- The labels (`TrainLabels.csv`) are not loaded yet, so there is nothing to fit against, and `fit()` is commented out.
- The four meta features are not appended yet. Windows are currently 11,400 values (57 channels: EEG + EOG).
- The array stacking after the loading loop raises an error: it calls `.shape` on a Python list, and it stacks per-file arrays that have different numbers of events.
- `hidden_layer_sizes=(11404, 1000, 1000, 2)` lists the input width and the two output classes as hidden layers. scikit-learn infers both, so this should be something like `(1000, 1000)`. As written, the first layer alone would have about 130 million weights.

## References

1. Perrin et al., *Objective and subjective evaluation of online error correction during P300-based spelling*, Advances in Human-Computer Interaction (2012). https://onlinelibrary.wiley.com/doi/10.1155/2012/578295
2. Winning solution by Alexandre Barachant and Rafał Cycoń: https://github.com/alexandrebarachant/bci-challenge-ner-2015
3. Lawhern et al., *EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces*, Journal of Neural Engineering (2018).
