# Semantic Book Recommendation System

This repository contains the full end-to-end pipeline for a semantic, description-based book recommendation system. It includes data cleaning, vector-based retrieval, category normalization, emotion analysis, and a lightweight Gradio dashboard for interactive exploration.

---

## Repository Structure

- `books_cleaned.csv`  
  Cleaned books dataset after preprocessing (missing value handling + description-length filtering).

- `books_with_categories.csv`  
  Enriched dataset with normalized **fiction / non-fiction** labels produced by the category classification step.

- `books_with_emotions.csv`  
  Enriched dataset with **emotion/tone tags** produced by the transformer-based emotion model.

- `notebook1_data_exploration.ipynb`  
  Dataset inspection and preprocessing workflow:
  - Missing-value analysis  
  - Description length filtering  
  - Feature engineering (e.g., tagged descriptions)

- `notebook2_vector_search.ipynb`  
  Semantic retrieval workflow:
  - Embedding generation (LLM embeddings)  
  - ChromaDB vector indexing  
  - Similarity search experiments and retrieval evaluation

- `notebook3_category_classification.ipynb`  
  Category normalization workflow:
  - Reference subset creation  
  - Zero-shot classification (fiction vs non-fiction)  
  - Precision-based evaluation and labeling across the catalog

- `notebook4_sentiment_analysis.ipynb`  
  Emotion/tone analysis workflow:
  - Transformer emotion inference  
  - Generating tone labels  
  - Connecting tone tags to recommendation filtering

- `tagged_description.txt`  
  Line-separated “ID + description” text used to build the vector store for retrieval.

- `recommender_dashboard.py`  
  Gradio application script that integrates:
  - Vector search results  
  - Category label display/filtering  
  - Emotion/tone display/filtering

- `cover-not-found.jpg`  
  Fallback image used when a book thumbnail is missing.

- `requirements.txt`  
  Python dependencies required to run notebooks and the Gradio dashboard.

- `README.md`  
  Project overview and instructions (edit as needed).

---

## How to Run (Quick Start)

1. Create and activate a Python environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Launch the dashboard:
   - `python recommender_dashboard.py`

---

## Notes

- Notebooks are organized to mirror the pipeline order: preprocessing → retrieval → category labeling → emotion tagging.
- Output CSVs are saved at each stage to keep the workflow modular and reproducible.
