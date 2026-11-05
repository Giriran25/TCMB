"""
TCMB - Static XLM-R Tokenization Baseline

Evaluates the original XLM-R tokenizer across eight languages
using the FLORES+ dataset.

Measurements:
1. Mean subword tokens per sentence.
2. Standalone SentencePiece marker frequency.
3. Marker token share.
4. Raw tokens per grapheme.
5. Content tokens per grapheme.
6. Content subwords per whitespace-separated word.

The standalone SentencePiece marker '▁' is excluded only
from content-fragmentation metrics, not from raw token counts.

FLORES+ is used for evaluation only. No trainable components
are fitted using this dataset.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

from evaluation.tokenization_metrics import (
    calculate_sentence_metrics,
    summarize_language_metrics,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_NAME = "FacebookAI/xlm-roberta-base"

LANGUAGES = {
    "eng_Latn": "English",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "kan_Knda": "Kannada",
    "mal_Mlym": "Malayalam",
    "hin_Deva": "Hindi",
    "cmn_Hans": "Chinese",
    "jpn_Jpan": "Japanese",
}


# ---------------------------------------------------------
# Evaluate one language
# ---------------------------------------------------------

def evaluate_language(
    tokenizer,
    marker_id,
    project_root,
    language_code,
    language_name,
    split_name,
    batch_size,
):
    """
    Evaluate static XLM-R tokenization for one language.

    Parameters
    ----------
    tokenizer:
        Pretrained XLM-R tokenizer.

    marker_id:
        Token ID corresponding to standalone '▁'.

    project_root:
        Root directory of the TCMB project.

    language_code:
        FLORES+ language configuration.

    language_name:
        Human-readable language name.

    split_name:
        Dataset split: dev or devtest.

    batch_size:
        Number of sentences processed per batch.

    Returns
    -------
    tuple:
        Language summary and sentence-level records.
    """

    dataset_path = (
        project_root
        / "data"
        / "flores_plus"
        / language_code
    )

    # Check whether the dataset exists.
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_path}"
        )

    print(f"\nLoading dataset: {language_code}")

    dataset_dict = load_from_disk(str(dataset_path))

    # Check split availability.
    if split_name not in dataset_dict:
        raise ValueError(
            f"Split '{split_name}' not found for {language_code}. "
            f"Available splits: {list(dataset_dict.keys())}"
        )

    dataset = dataset_dict[split_name]

    # Check required columns.
    required_columns = {"id", "text"}

    missing_columns = required_columns - set(
        dataset.column_names
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(
        f"\nEvaluating {language_name} "
        f"({language_code}) - {split_name}"
    )

    sentence_records = []

    # -----------------------------------------------------
    # Process dataset in batches
    # -----------------------------------------------------

    for start in tqdm(
        range(0, len(dataset), batch_size),
        desc=language_name,
    ):

        end = min(start + batch_size, len(dataset))

        batch = dataset[start:end]

        texts = list(batch["text"])
        sentence_ids = list(batch["id"])

        # Tokenize without BOS/EOS special tokens.
        # This isolates the subword tokenization behaviour.
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )

        input_ids_batch = encoded["input_ids"]

        # -------------------------------------------------
        # Calculate metrics for each sentence
        # -------------------------------------------------

        for sentence_id, text, input_ids in zip(
            sentence_ids,
            texts,
            input_ids_batch,
        ):

            # Total subword token count.
            token_count = len(input_ids)

            # Count standalone SentencePiece markers.
            marker_count = input_ids.count(marker_id)

            # Calculate sentence-level metrics.
            metrics = calculate_sentence_metrics(
                text=text,
                token_count=token_count,
                marker_count=marker_count,
                language_code=language_code,
            )

            sentence_records.append({
                "language_code": language_code,
                "language": language_name,
                "split": split_name,
                "sentence_id": sentence_id,
                **metrics,
            })

    # -----------------------------------------------------
    # Aggregate language-level results
    # -----------------------------------------------------

    summary = summarize_language_metrics(
        sentence_records=sentence_records,
        language_code=language_code,
        language_name=language_name,
        split_name=split_name,
    )

    print(
        f"Completed {language_name}: "
        f"{len(sentence_records)} sentences"
    )

    return summary, sentence_records


# ---------------------------------------------------------
# Generate evaluation figures
# ---------------------------------------------------------

def generate_figures(
    summary_df,
    figures_dir,
    split_name,
):
    """
    Generate baseline tokenization figures.

    Figure 1:
        Content subword tokens per grapheme.

    Figure 2:
        Content subwords per whitespace-separated word.

    Figure 3:
        Standalone SentencePiece marker share.
    """

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Figure 1: Content tokens per grapheme
    # -----------------------------------------------------

    ordered = summary_df.sort_values(
        "content_tokens_per_grapheme",
        ascending=False,
    )

    plt.figure(figsize=(11, 6))

    plt.bar(
        ordered["language"],
        ordered["content_tokens_per_grapheme"],
    )

    plt.title(
        "Static XLM-R: Content Subword Pieces per Grapheme "
        f"({split_name})"
    )

    plt.xlabel("Language")

    plt.ylabel(
        "Content Subword Pieces per Grapheme"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    path1 = (
        figures_dir
        / f"static_xlmr_content_tokens_per_grapheme_{split_name}.png"
    )

    plt.savefig(
        path1,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------
    # Figure 2: Content subwords per whitespace word
    # -----------------------------------------------------

    fertility_df = summary_df.dropna(
        subset=[
            "content_subwords_per_whitespace_word"
        ]
    ).sort_values(
        "content_subwords_per_whitespace_word",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        fertility_df["language"],
        fertility_df[
            "content_subwords_per_whitespace_word"
        ],
    )

    plt.title(
        "Static XLM-R: Content Subwords per "
        f"Whitespace Word ({split_name})"
    )

    plt.xlabel("Language")

    plt.ylabel(
        "Content Subwords per Whitespace Word"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    path2 = (
        figures_dir
        / f"static_xlmr_content_whitespace_fertility_{split_name}.png"
    )

    plt.savefig(
        path2,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------
    # Figure 3: Standalone marker token share
    # -----------------------------------------------------

    marker_df = summary_df.sort_values(
        "marker_token_share",
        ascending=False,
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        marker_df["language"],
        marker_df["marker_token_share"],
    )

    plt.title(
        "Static XLM-R: Standalone SentencePiece "
        f"Marker Share ({split_name})"
    )

    plt.xlabel("Language")

    plt.ylabel("Marker Token Share")

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    path3 = (
        figures_dir
        / f"static_xlmr_marker_share_{split_name}.png"
    )

    plt.savefig(
        path3,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -----------------------------------------------------
    # Display saved figures
    # -----------------------------------------------------

    print("\nFigures saved successfully:")

    print(path1)
    print(path2)
    print(path3)


# ---------------------------------------------------------
# Main execution
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate static XLM-R tokenization "
            "across eight FLORES+ languages."
        )
    )

    parser.add_argument(
        "--split",
        choices=["dev", "devtest"],
        default="dev",
        help="FLORES+ dataset split.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sentences processed per batch.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "Batch size must be at least 1."
        )

    # -----------------------------------------------------
    # Project directories
    # -----------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    tables_dir = (
        project_root
        / "results"
        / "tables"
    )

    figures_dir = (
        project_root
        / "results"
        / "figures"
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Display experiment information
    # -----------------------------------------------------

    print("=" * 70)

    print(
        "TCMB - STATIC XLM-R TOKENIZATION BASELINE"
    )

    print("=" * 70)

    print(f"Model: {MODEL_NAME}")

    print(f"Dataset split: {args.split}")

    print(f"Batch size: {args.batch_size}")

    # -----------------------------------------------------
    # Load tokenizer
    # -----------------------------------------------------

    print("\nLoading XLM-R tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print(
        "Tokenizer loaded successfully."
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size,
    )

    # Standalone SentencePiece marker ID.
    marker_id = tokenizer.convert_tokens_to_ids(
        "▁"
    )

    print(
        "Standalone marker:",
        tokenizer.convert_ids_to_tokens([marker_id]),
    )

    print(
        "Standalone marker ID:",
        marker_id,
    )

    # -----------------------------------------------------
    # Evaluate all eight languages
    # -----------------------------------------------------

    all_summaries = []

    all_sentence_records = []

    for language_code, language_name in LANGUAGES.items():

        summary, sentence_records = evaluate_language(
            tokenizer=tokenizer,
            marker_id=marker_id,
            project_root=project_root,
            language_code=language_code,
            language_name=language_name,
            split_name=args.split,
            batch_size=args.batch_size,
        )

        all_summaries.append(summary)

        all_sentence_records.extend(
            sentence_records
        )

    # -----------------------------------------------------
    # Convert results into DataFrames
    # -----------------------------------------------------

    summary_df = pd.DataFrame(
        all_summaries
    )

    sentence_df = pd.DataFrame(
        all_sentence_records
    )

    # -----------------------------------------------------
    # Save CSV files
    # -----------------------------------------------------

    summary_path = (
        tables_dir
        / f"static_xlmr_summary_{args.split}.csv"
    )

    sentence_path = (
        tables_dir
        / f"static_xlmr_sentence_metrics_{args.split}.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    sentence_df.to_csv(
        sentence_path,
        index=False,
    )

    print("\nCSV files saved successfully:")

    print(summary_path)

    print(sentence_path)

    # -----------------------------------------------------
    # Generate figures
    # -----------------------------------------------------

    generate_figures(
        summary_df=summary_df,
        figures_dir=figures_dir,
        split_name=args.split,
    )

    # -----------------------------------------------------
    # Display final summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "STATIC XLM-R BASELINE COMPLETED"
    )

    print("=" * 70)

    display_columns = [
        "language",
        "sentence_count",
        "mean_subwords_per_sentence",
        "markers_per_sentence",
        "marker_token_share",
        "raw_tokens_per_grapheme",
        "content_tokens_per_grapheme",
        "content_subwords_per_whitespace_word",
    ]

    print("\nLanguage-wise summary:\n")

    print(
        summary_df[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # -----------------------------------------------------
    # Final statistics
    # -----------------------------------------------------

    total_sentences = len(sentence_df)

    total_languages = len(summary_df)

    print("\nExperiment statistics:")

    print(
        "Languages evaluated:",
        total_languages,
    )

    print(
        "Total sentences evaluated:",
        total_sentences,
    )

    print(
        "Dataset split:",
        args.split,
    )

    print("\nAll result files are available under:")

    print(project_root / "results")

    print("\nBaseline evaluation finished successfully.")


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()