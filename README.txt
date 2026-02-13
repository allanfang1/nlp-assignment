# Natural Language Processing: Word Embeddings and Multilingual Alignment

A comprehensive NLP project demonstrating various word embedding techniques, multilingual embedding alignment, and a downstream language identification task using English-French parallel corpora.

## Project Overview

This project implements and compares multiple word embedding approaches:
- **One-Hot Encoding**: Simple sparse representation
- **TF-IDF**: Term frequency weighted by inverse document frequency
- **Word2Vec**: Neural network-based dense embeddings
- **GloVe**: Pre-trained embeddings from global word co-occurrence
- **FastText**: Subword-aware embeddings for handling OOV words

Additionally:
- Multilingual embedding alignment using orthogonal Procrustes
- Visualization of semantic relationships using t-SNE
- Synonym/antonym detection
- Polysemy analysis
- Cross-lingual word similarity
- Language identification classifier

## Project Structure
- nlp.ipynb # Main Jupyter notebook with all implementations
- sentencepairs.tsv # English-French parallel sentence pairs (required - to download)
- README.md # This file

## Prerequisites

### Required Libraries

Install all required packages using pip:

pip install pandas numpy scikit-learn gensim matplotlib seaborn scipy

### Python Version
Python 3.7 or higher recommended

### Data Requirements
Primary Dataset: sentencepairs.tsv
The notebook expects a tab-separated file with English-French sentence pairs:

Columns:

- src_id: Source sentence identifier
- english: English sentence
- tgt_id: Target sentence identifier
- french: French translation

Path: Update the file path in the notebook

df = pd.read_csv("path/to/your/sentencepairs.tsv", sep="\t", header=None, 
                 names=["src_id", "english", "tgt_id", "french"])

## Run The Code

- Upload notebook to Google Colab
- Mount Google Drive if data is stored there
- Execute Cells in Order (run cells sequentially as they have dependencies)
