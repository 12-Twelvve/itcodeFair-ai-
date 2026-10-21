"""
Loads the two-stage-trained model and serves exactly what the brief asks
for: accepts a product/laptop name (+ its reviews), produces an
aspect-sentiment table. Judges spend 4-5 minutes here, so model stats and
the key output are both visible immediately, no digging required.

Usage:
    streamlit run final_app.py

Expects these files in the same directory:
    two_stage_checkpoint.pt
    aspect_category_vocab.json
    thresholds.npy
    test_results_two_stage.json
"""

import json
import time
import os
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "microsoft/deberta-v3-base"
POLARITIES = ["positive", "negative", "neutral"]
MAX_LENGTH = 384
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = "two_stage_checkpoint.pt"
VOCAB_PATH = "aspect_category_vocab.json"
THRESHOLDS_PATH = "thresholds.npy"
TEST_RESULTS_PATH = "test_results.json"

st.set_page_config(page_title="ABSA — Aspect Sentiment Explorer", layout="wide")


# ---------------------------------------------------------------------------
# Model definition (must match training script exactly)
# ---------------------------------------------------------------------------

class JointABSAModel(nn.Module):
    def __init__(self, model_name, num_categories, num_polarities=3, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden, num_categories)
        self.polarity_head = nn.Linear(hidden, num_categories * num_polarities)
        self.num_categories = num_categories
        self.num_polarities = num_polarities

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0])
        cat_logits = self.category_head(pooled)
        pol_logits = self.polarity_head(pooled).view(-1, self.num_categories, self.num_polarities)
        return cat_logits, pol_logits


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(VOCAB_PATH) and os.path.exists(THRESHOLDS_PATH)):
        return None, None, None, None

    with open(VOCAB_PATH) as f:
        category_vocab = json.load(f)
    thresholds = np.load(THRESHOLDS_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = JointABSAModel(MODEL_NAME, num_categories=len(category_vocab)).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    return model, tokenizer, category_vocab, thresholds


def predict_absa(text, model, tokenizer, category_vocab, thresholds):
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH,
                     return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        cat_logits, pol_logits = model(enc["input_ids"], enc["attention_mask"])
    cat_probs = torch.sigmoid(cat_logits)[0].cpu().numpy()
    pol_preds = pol_logits[0].argmax(-1).cpu().numpy()

    results = []
    for i, cat in enumerate(category_vocab):
        if cat_probs[i] >= thresholds[i]:
            results.append({
                "category": cat,
                "polarity": POLARITIES[pol_preds[i]],
                "confidence": float(cat_probs[i]),
            })
    return sorted(results, key=lambda x: -x["confidence"])


def polarity_to_stars(pos, neg, neu):
    """Net-sentiment score mapped to 1-5 stars, matching the brief's mockup."""
    total = pos + neg + neu
    if total == 0:
        return 0
    score = (pos - neg) / total  # -1..1
    stars = round(3 + score * 2)  # maps -1->1 star, 0->3 stars, 1->5 stars
    return max(1, min(5, stars))


def star_string(n):
    return "\u2b50" * n + "\u2606" * (5 - n) if n > 0 else "\u2014"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("Aspect-Based Sentiment Analysis")
st.caption("CDU IT Code Fair 2026 — enter a product name and its reviews to generate an aspect-sentiment table")

model, tokenizer, category_vocab, thresholds = load_model()

with st.sidebar:
    st.header("Model stats")
    if model is None:
        st.error("Model artifacts not found. Run 06_two_stage_training.py first "
                  "and place the checkpoint, vocab, and thresholds files alongside this app.")
    else:
        st.success(f"Loaded — {len(category_vocab)} categories")
        if os.path.exists(TEST_RESULTS_PATH):
            with open(TEST_RESULTS_PATH) as f:
                stats = json.load(f)
            st.metric("Test F1 (exact-match tuple)", f"{stats.get('f1', 0):.3f}")
            st.metric("Precision / Recall", f"{stats.get('precision', 0):.3f} / {stats.get('recall', 0):.3f}")
            st.metric("Latency", f"{stats.get('ms_per_review', 0):.0f} ms/review")
            st.metric("Peak memory", f"{stats.get('peak_memory_mb', 0):.0f} MB")
        else:
            st.info("Run the test evaluation in 06_two_stage_training.py to populate these stats.")

DEMO_REVIEWS = (
    "The so called laptop runs too slow and I hate it!\n\n"
    "Great battery life but the screen is dim and the keyboard feels cheap.\n\n"
    "As a lifelong Windows user, I was extremely pleased with the switch. "
    "Setup was fast and customer service was very helpful when I had a question.\n\n"
    "It is the worst laptop ever. Do not buy it!"
)

col1, col2 = st.columns([1, 2])
with col1:
    product_name = st.text_input("Product / laptop name", placeholder="e.g. Apple Mac mini")
with col2:
    st.caption("Paste one review per paragraph (separate with a blank line).")

reviews_text = st.text_area("Reviews", value=DEMO_REVIEWS, height=180)

run = st.button("Analyze", type="primary", disabled=(model is None))

if run:
    reviews = [r.strip() for r in reviews_text.split("\n\n") if r.strip()]
    if not reviews:
        st.warning("Enter at least one review.")
    else:
        t0 = time.perf_counter()
        all_rows = []
        for i, review in enumerate(reviews):
            preds = predict_absa(review, model, tokenizer, category_vocab, thresholds)
            for p in preds:
                all_rows.append({"Review #": i + 1, "Category": p["category"],
                                  "Polarity": p["polarity"], "Confidence": round(p["confidence"], 2)})
        elapsed = time.perf_counter() - t0

        title = f"Aspect-sentiment table — {product_name}" if product_name else "Aspect-sentiment table"
        st.subheader(title)
        st.caption(f"Analyzed {len(reviews)} review(s) in {elapsed*1000:.0f} ms "
                   f"({elapsed/len(reviews)*1000:.1f} ms/review)")

        df = pd.DataFrame(all_rows)

        if df.empty:
            st.info("No aspects detected above the confidence threshold for these reviews.")
        else:
            # --- Aggregated summary (the star-rating table from the brief's mockup) ---
            st.markdown("#### Summary across all reviews")
            summary_rows = []
            for cat in sorted(df["Category"].unique()):
                cat_df = df[df["Category"] == cat]
                pos = (cat_df["Polarity"] == "positive").sum()
                neg = (cat_df["Polarity"] == "negative").sum()
                neu = (cat_df["Polarity"] == "neutral").sum()
                stars = polarity_to_stars(pos, neg, neu)
                summary_rows.append({
                    "Aspect": cat, "Mentions": len(cat_df),
                    "Positive": pos, "Negative": neg, "Neutral": neu,
                    "Rating": star_string(stars),
                })
            summary_df = pd.DataFrame(summary_rows).sort_values("Mentions", ascending=False)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # --- Full detail table ---
            with st.expander("Per-review detail"):
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"absa_results_{product_name or 'product'}.csv",
                mime="text/csv",
            )
