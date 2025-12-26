import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import gradio as gr

load_dotenv()

# ---------- Load Models & Data ----------

embedding_model_path = "/Users/mukundkomati/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

books_df = pd.read_csv("books_with_emotions.csv")

books_df["large_thumbnail"] = books_df["thumbnail"] + "&fife=w800"
books_df["large_thumbnail"] = np.where(
    books_df["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books_df["large_thumbnail"],
)

loaded_documents = TextLoader("tagged_description.txt").load()
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=0,
    chunk_overlap=0
)
split_documents = text_splitter.split_documents(loaded_documents)

embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

books_vector_store = Chroma.from_documents(
    split_documents,
    embedding_model
)

# Global variable to store recommendations
latest_recommendations = None


# ---------- Semantic Retrieval ----------

def retrieve_semantic_recommendations(
        user_query: str,
        selected_category: str = None,
        selected_tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
):
    retrieved_docs = books_vector_store.similarity_search(
        user_query,
        k=initial_top_k
    )

    retrieved_isbns = [
        int(doc.page_content.strip('"').split()[0])
        for doc in retrieved_docs
    ]

    candidate_books = (
        books_df
        [books_df["isbn13"].isin(retrieved_isbns)]
        .head(initial_top_k)
    )

    if selected_category != "All":
        candidate_books = (
            candidate_books
            [candidate_books["simple_categories"] == selected_category]
            .head(final_top_k)
        )
    else:
        candidate_books = candidate_books.head(final_top_k)

    if selected_tone == "Happy":
        candidate_books.sort_values(by="joy", ascending=False, inplace=True)
    elif selected_tone == "Surprising":
        candidate_books.sort_values(by="surprise", ascending=False, inplace=True)
    elif selected_tone == "Angry":
        candidate_books.sort_values(by="anger", ascending=False, inplace=True)
    elif selected_tone == "Suspenseful":
        candidate_books.sort_values(by="fear", ascending=False, inplace=True)
    elif selected_tone == "Sad":
        candidate_books.sort_values(by="sadness", ascending=False, inplace=True)

    return candidate_books


# ---------- Build Result List for Gallery ----------

def recommend_books(user_query: str, selected_category: str, selected_tone: str):
    global latest_recommendations

    latest_recommendations = retrieve_semantic_recommendations(
        user_query,
        selected_category,
        selected_tone
    )

    gallery_results = []

    for _, book_row in latest_recommendations.iterrows():
        description_text = book_row["description"]
        truncated_description = " ".join(
            description_text.split()[:30]
        ) + "..."

        author_list = book_row["authors"].split(";")
        if len(author_list) == 2:
            authors_display = f"{author_list[0]} and {author_list[1]}"
        elif len(author_list) > 2:
            authors_display = f"{', '.join(author_list[:-1])}, and {author_list[-1]}"
        else:
            authors_display = book_row["authors"]

        caption_text = (
            f"{book_row['title']} by {authors_display}: "
            f"{truncated_description}"
        )

        gallery_results.append(
            (book_row["large_thumbnail"], caption_text)
        )

    return gallery_results


# ---------- Click Handler for Full Description (HTML) ----------

def show_full_description(event: gr.SelectData):
    selected_index = event.index

    if latest_recommendations is None:
        return "<div style='font-size:20px;'>No recommendations available.</div>"

    selected_book = latest_recommendations.iloc[selected_index]
    authors_display = selected_book["authors"].replace(";", ", ")

    full_description_html = f"""
<div style="font-size:20px; line-height:1.7; padding:12px;">
  <h2 style="font-size:30px; margin-bottom:10px;">{selected_book['title']}</h2>

  <div style="font-size:20px; margin-bottom:12px;">
    <b>Authors:</b> {authors_display}<br>
    <b>Category:</b> {selected_book['simple_categories']}
  </div>

  <hr style="margin:15px 0;">

  <h3 style="font-size:24px; margin-bottom:10px;">Full Description</h3>

  <div style="font-size:20px;">
    {selected_book['description']}
  </div>
</div>
"""
    return full_description_html


# ---------- Gradio UI ----------

available_categories = ["All"] + sorted(
    books_df["simple_categories"].unique()
)
available_tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(
    theme=gr.themes.Glass(),
    css="""
    .gradio-container {
        font-size: 20px;
    }

    #app-title {
        font-size: 150px;
        font-weight: 900;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 6px;
        letter-spacing: 0.01em;
        background: linear-gradient(90deg, #8ab4ff, #d8b4fe, #f9a8d4);
        -webkit-background-clip: text;
        color: transparent;
        filter: drop-shadow(0px 2px 4px rgba(255,255,255,0.25));
    }

    #app-subtitle {
        text-align: center;
        font-size: 22px;
        color: #d1d5db;
        margin-bottom: 25px;
    }

    #recommendations-title {
        font-size: 30px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }

    .gr-textbox label,
    .gr-dropdown label {
        font-size: 20px;
    }

    #find-button {
        font-size: 22px !important;
        font-weight: 700 !important;
        padding: 0.85rem 2rem !important;
        border-radius: 999px !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #ec4899) !important;
        color: white !important;
        border: none !important;
        cursor: pointer;
        transition: all 0.16s ease-out;
        box-shadow: 0 8px 24px rgba(96, 165, 250, 0.35);
    }

    #find-button:hover {
        transform: translateY(-1px) scale(1.015);
        box-shadow: 0 12px 28px rgba(147, 51, 234, 0.45);
        opacity: 0.96;
    }

    #find-button:active {
        transform: translateY(1px) scale(0.985);
        opacity: 0.88;
    }

    .gr-gallery .thumbnail-item .caption,
    .gr-gallery .gallery-item > div:last-child {
        font-size: 18px !important;
    }
    """
) as dashboard:

    gr.Markdown("# 📚 Semantic Book Recommender", elem_id="app-title")
    gr.Markdown(
        "Find books by <i>vibe</i>, <i>theme</i>, and <i>emotion</i>.",
        elem_id="app-subtitle"
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=available_categories,
            label="Select a category:",
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=available_tones,
            label="Select an emotional tone:",
            value="All"
        )
        search_button = gr.Button(
            "🔍 Find recommendations",
            elem_id="find-button"
        )

    gr.Markdown("## Recommendations", elem_id="recommendations-title")
    recommendations_gallery = gr.Gallery(
        label="Recommended Books",
        columns=8,
        rows=2
    )

    book_details_html = gr.HTML(
        "<div style='font-size:20px;'>Click on a book to see full details</div>",
        elem_id="book-details"
    )

    search_button.click(
        fn=recommend_books,
        inputs=[query_input, category_dropdown, tone_dropdown],
        outputs=recommendations_gallery
    )

    recommendations_gallery.select(
        fn=show_full_description,
        inputs=None,
        outputs=book_details_html
    )


if __name__ == "__main__":
    dashboard.launch()
