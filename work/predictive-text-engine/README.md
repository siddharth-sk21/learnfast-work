# N-gram Predictive Text Engine

A Python-based predictive text engine that learns word patterns from literary text and predicts likely next words using Bigram and Trigram language models.

The project implements N-gram frequency counting, Add-k smoothing, next-word prediction, probabilistic text generation, and perplexity-based evaluation.

## Project Overview

Predictive text systems estimate the next word based on previously observed words. This project implements two statistical language models from scratch:

- **Bigram Model:** Predicts the next word using the previous one.
- **Trigram Model:** Predicts the next word using the previous two.

The models are trained on a collection of public-domain literary texts from the NLTK Gutenberg corpus.

## Features

- Text tokenization and preprocessing
- Bigram and Trigram frequency counting
- Add-k smoothing to handle unseen n-grams
- Next-word suggestions
- Probabilistic text generation
- Perplexity-based model evaluation
- Validation-based smoothing parameter selection
- Results saved in CSV format
- Visualization of experimental results

## Tech Stack

- Python
- NLTK Gutenberg Corpus
- Matplotlib
- Collections (`Counter`)
- Regular Expressions
- CSV

## Project Structure

```text
predictive-text-engine/
│
├── data/
│   ├── corpus.txt
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
│
├── results/
│   ├── final_comparison.csv
│   └── final_comparison.png
│
├── 01_data_preparation.ipynb
├── ngram_lm.py
├── main.py
├── plot_results.py
└── README.md
```

## Dataset

The training corpus was assembled from seven public-domain books available through the NLTK Gutenberg collection.

The dataset was divided at the sentence level, with each selected book represented in all three splits.

| Dataset | Sentences | Split |
|---|---:|---:|
| Training | 25,295 | 80% |
| Validation | 3,162 | 10% |
| Testing | 3,166 | 10% |
| **Total** | **31,623** | **100%** |

A fixed random seed (`42`) was used for reproducibility.

## Model Architecture

### 1. N-gram Frequency Counting

The model counts occurrences of word sequences.

For example, the sentence:

`emma was quite happy`

produces these bigrams:

- (emma, was)
- (was, quite)
- (quite, happy)

A trigram model additionally considers three-word sequences.

### 2. Add-k Smoothing

Add-k smoothing assigns nonzero probabilities to unseen n-grams.

\[
P(w \mid context) =
\frac{C(context,w)+k}
{C(context)+kV}
\]

Where:

- \(C(context,w)\): Count of the n-gram.
- \(C(context)\): Count of the context.
- \(k\): Smoothing parameter.
- \(V\): Vocabulary size.

The experiment compares three smoothing values: `0.1`, `0.5`, and `1.0`.

### 3. Next-word Prediction

Given a context, the model calculates probabilities for candidate words and returns the top-k suggestions.

### 4. Text Generation

The engine generates text iteratively by sampling from the most probable next-word candidates.

### 5. Perplexity Evaluation

Perplexity measures how well a language model predicts unseen text. Lower perplexity indicates better predictive performance on the evaluated dataset.

## Experimental Results

The smoothing parameter was selected using validation data. The final models were then evaluated on the untouched test dataset.

| Metric | Bigram | Trigram |
|---|---:|---:|
| Best smoothing parameter | 0.1 | 0.1 |
| Validation perplexity | **290.42** | 1,727.36 |
| Test perplexity | **290.13** | 1,720.43 |
| Unique n-grams | 201,391 | 469,369 |

### Key Observations

1. The Bigram model achieved lower perplexity than the Trigram model on this dataset.
2. A smoothing value of `0.1` performed best among the tested configurations for both models.
3. The Trigram model produced substantially more unique n-grams, increasing the impact of data sparsity.
4. The validation and test perplexities were relatively close, suggesting consistent performance across the two splits.

These findings are specific to the selected corpus, preprocessing, and model configurations. They do not establish that Bigram models are universally superior to Trigram models.

## How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd predictive-text-engine
```

Ensure Python is installed and the dataset files are present inside `data/`.

Run the model comparison:

```bash
python main.py
```

Generate the final comparison graph:

```bash
pip install matplotlib
python plot_results.py
```

The final experimental results are saved in:

```text
results/final_comparison.csv
results/final_comparison.png
```

## Limitations

- N-gram models use a limited context and cannot understand long-range dependencies.
- Word predictions depend heavily on the training corpus.
- Text generation can become repetitive or grammatically inconsistent.
- Trigram models experience greater data sparsity than bigram models in this experiment.
- Add-k smoothing does not capture deeper linguistic relationships.

## Future Improvements

- Implement backoff and interpolation techniques.
- Experiment with larger and more diverse datasets.
- Compare against a neural language model.
- Build a web-based autocomplete interface.
- Improve sentence generation with more advanced sampling strategies.

## Learning Outcomes

Through this project, I explored statistical language modelling, probability estimation, smoothing techniques, text generation, and evaluation using perplexity.

The project also provided practical experience in designing reproducible experiments and comparing different model configurations.

---

**Built with Python | Statistical NLP | Language Modelling**# N-gram Predictive Text Engine

A Python-based predictive text engine that learns word patterns from literary text and predicts likely next words using Bigram and Trigram language models.

The project implements N-gram frequency counting, Add-k smoothing, next-word prediction, probabilistic text generation, and perplexity-based evaluation.

## Project Overview

Predictive text systems estimate the next word based on previously observed words. This project implements two statistical language models from scratch:

- **Bigram Model:** Predicts the next word using the previous one.
- **Trigram Model:** Predicts the next word using the previous two.

The models are trained on a collection of public-domain literary texts from the NLTK Gutenberg corpus.

## Features

- Text tokenization and preprocessing
- Bigram and Trigram frequency counting
- Add-k smoothing to handle unseen n-grams
- Next-word suggestions
- Probabilistic text generation
- Perplexity-based model evaluation
- Validation-based smoothing parameter selection
- Results saved in CSV format
- Visualization of experimental results

## Tech Stack

- Python
- NLTK Gutenberg Corpus
- Matplotlib
- Collections (`Counter`)
- Regular Expressions
- CSV

## Project Structure

```text
predictive-text-engine/
│
├── data/
│   ├── corpus.txt
│   ├── train.txt
│   ├── validation.txt
│   └── test.txt
│
├── results/
│   ├── final_comparison.csv
│   └── final_comparison.png
│
├── 01_data_preparation.ipynb
├── ngram_lm.py
├── main.py
├── plot_results.py
└── README.md
```

## Dataset

The training corpus was assembled from seven public-domain books available through the NLTK Gutenberg collection.

The dataset was divided at the sentence level, with each selected book represented in all three splits.

| Dataset | Sentences | Split |
|---|---:|---:|
| Training | 25,295 | 80% |
| Validation | 3,162 | 10% |
| Testing | 3,166 | 10% |
| **Total** | **31,623** | **100%** |

A fixed random seed (`42`) was used for reproducibility.

## Model Architecture

### 1. N-gram Frequency Counting

The model counts occurrences of word sequences.

For example, the sentence:

`emma was quite happy`

produces these bigrams:

- (emma, was)
- (was, quite)
- (quite, happy)

A trigram model additionally considers three-word sequences.

### 2. Add-k Smoothing

Add-k smoothing assigns nonzero probabilities to unseen n-grams.

\[
P(w \mid context) =
\frac{C(context,w)+k}
{C(context)+kV}
\]

Where:

- \(C(context,w)\): Count of the n-gram.
- \(C(context)\): Count of the context.
- \(k\): Smoothing parameter.
- \(V\): Vocabulary size.

The experiment compares three smoothing values: `0.1`, `0.5`, and `1.0`.

### 3. Next-word Prediction

Given a context, the model calculates probabilities for candidate words and returns the top-k suggestions.

### 4. Text Generation

The engine generates text iteratively by sampling from the most probable next-word candidates.

### 5. Perplexity Evaluation

Perplexity measures how well a language model predicts unseen text. Lower perplexity indicates better predictive performance on the evaluated dataset.

## Experimental Results

The smoothing parameter was selected using validation data. The final models were then evaluated on the untouched test dataset.

| Metric | Bigram | Trigram |
|---|---:|---:|
| Best smoothing parameter | 0.1 | 0.1 |
| Validation perplexity | **290.42** | 1,727.36 |
| Test perplexity | **290.13** | 1,720.43 |
| Unique n-grams | 201,391 | 469,369 |

### Key Observations

1. The Bigram model achieved lower perplexity than the Trigram model on this dataset.
2. A smoothing value of `0.1` performed best among the tested configurations for both models.
3. The Trigram model produced substantially more unique n-grams, increasing the impact of data sparsity.
4. The validation and test perplexities were relatively close, suggesting consistent performance across the two splits.

These findings are specific to the selected corpus, preprocessing, and model configurations. They do not establish that Bigram models are universally superior to Trigram models.

## How to Run

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd predictive-text-engine
```

Ensure Python is installed and the dataset files are present inside `data/`.

Run the model comparison:

```bash
python main.py
```

Generate the final comparison graph:

```bash
pip install matplotlib
python plot_results.py
```

The final experimental results are saved in:

```text
results/final_comparison.csv
results/final_comparison.png
```

## Limitations

- N-gram models use a limited context and cannot understand long-range dependencies.
- Word predictions depend heavily on the training corpus.
- Text generation can become repetitive or grammatically inconsistent.
- Trigram models experience greater data sparsity than bigram models in this experiment.
- Add-k smoothing does not capture deeper linguistic relationships.

## Future Improvements

- Implement backoff and interpolation techniques.
- Experiment with larger and more diverse datasets.
- Compare against a neural language model.
- Build a web-based autocomplete interface.
- Improve sentence generation with more advanced sampling strategies.

## Learning Outcomes

Through this project, I explored statistical language modelling, probability estimation, smoothing techniques, text generation, and evaluation using perplexity.

The project also provided practical experience in designing reproducible experiments and comparing different model configurations.

---

**Built with Python | Statistical NLP | Language Modelling**