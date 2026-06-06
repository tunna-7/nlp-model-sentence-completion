# Neural Language Model for Sentence Completion

## Instructions

1. Follow the instructions in each question carefully.
2. A Jupyter notebook in PDF format along with output for each cell is expected.
3. Any assignment submitted using other Python IDEs will not be considered for grading.
4. Incorrect Assignment Set submitted will not be considered.
5. Observation and inference have to be mentioned for each question. Techniques used without justification or inference will not be awarded marks.

---

# 1. Objective & Context

Language modelling is a foundational task in NLP, where the goal is to predict the next word in a sequence given the preceding words.

In this assignment, you will build and train a character-level or word-level Neural Language Model to perform a sentence completion task.

Given a partial prompt (e.g., `"The weather today is exceptionally..."`), your model must predict the most likely next word(s) to complete the sentence coherently.

---

# 2. Dataset Requirements

You may use any standard, publicly available text dataset, such as:

* WikiText-2 dataset
* Shakespeare’s plays
* A subset of Project Gutenberg books

Dataset Reference:
https://docs.pytorch.org/text/0.8.1/datasets.html

Ensure your text is:

* Cleaned
* Tokenized
* Split into:

  * Training Set: 80%
  * Validation Set: 10%
  * Testing Set: 10%

---

# 3. Tasks and Mark Breakdown

## Task 1: Implement the Below Tasks Using Virtual Lab (1 Mark)

---

## Task 2: Data Pre-processing & Pipeline Building (3 Marks)

### 1. Tokenization & Vocabulary (1 Mark)

* Clean the text data:

  * Lowercasing
  * Handling punctuation
* Build a vocabulary dictionary mapping tokens to unique integers.
* Handle Out-Of-Vocabulary (OOV) tokens using an `<UNK>` token.

### 2. Dataset Generation (2 Marks)

Create a sliding window data generator.

For a chosen sequence length `N`:

* **Input:**
  `[w1, w2, w3, ..., wN]`

* **Target:**
  `w(N+1)`

Batch the data efficiently for training.

---

## Task 3: Model Architecture Implementation (3 Marks)

Implement a recurrent neural network using a deep learning framework:

* PyTorch
* TensorFlow

### 1. Embedding Layer (1 Mark)

Map input token IDs into a dense continuous vector space of size `D`.

### 2. Recurrent Layer (1 Mark)

Implement either:

* LSTM
* GRU

Requirements:

* At least 128 hidden units
* Capture sequential dependencies

### 3. Output Layer (1 Mark)

Implement:

* A Linear layer
* Followed by Softmax activation

Output:

* Probability distribution over the vocabulary size `V`

---

## Task 4: Training & Evaluation (3 Marks)

### 1. Loss Function & Optimization (1 Mark)

Train the model using:

* Cross-Entropy Loss
* Optimizer such as Adam

### 2. Perplexity (1 Mark)

Track and plot:

* Training loss
* Validation loss

Compute and report final Perplexity (PPL) on the test set.

Perplexity formula:

[
PPL = \exp(\text{Cross-Entropy Loss})
]

### 3. Hyperparameter Tuning (1 Mark)

Demonstrate the impact of tweaking at least one hyperparameter, such as:

* Embedding size
* Learning rate
* Sequence length

Analyze its effect on validation performance.

---

## Task 5: Sentence Completion Inference (2 Marks)

### 1. Greedy Search vs. Sampling

Write an inference function that:

* Takes an arbitrary prompt string
* Tokenizes the input
* Feeds it into the trained model
* Generates the next 5 words

### 2. Compare Decoding Strategies

#### Greedy Decoding

* Pick the token with the highest probability.

#### Temperature-Scaled Sampling

* Divide logits by temperature `T` before applying Softmax.

Effects of Temperature:

* Higher `T`:

  * Flatter probability distribution
  * More random / creative outputs

* Lower `T`:

  * Sharper distribution
  * More deterministic outputs

---

# Important Note

**Justification of the output obtained for all the above tasks is mandatory.**
