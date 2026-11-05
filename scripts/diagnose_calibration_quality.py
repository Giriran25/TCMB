"""
TCMB - Randomized Calibration Corpus Quality Diagnostics

Inspects 200 shuffled documents per language.

Sources:
    English: FineWeb sample-10BT
    Other languages: FineWeb-2

The script:
    - Uses streaming.
    - Shuffles records using a fixed seed.
    - Measures document length.
    - Examines language identification scores.
    - Measures duplicate cluster statistics.
    - Examines domain diversity.
    - Saves metadata only, not document text.

No model training is performed.
"""

from datasets import load_dataset
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


# =====================================================
# CONFIGURATION
# =====================================================

SOURCES = {
    "English": {
        "repository": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
    },
    "Tamil": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "tam_Taml",
    },
    "Telugu": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "tel_Telu",
    },
    "Kannada": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "kan_Knda",
    },
    "Malayalam": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "mal_Mlym",
    },
    "Hindi": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "hin_Deva",
    },
    "Chinese": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "cmn_Hani",
    },
    "Japanese": {
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "jpn_Jpan",
    },
}

SAMPLES_PER_LANGUAGE = 200

SHUFFLE_BUFFER_SIZE = 10000

RANDOM_SEED = 42


# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_domain(url):

    if not isinstance(url, str) or not url.strip():
        return None

    try:
        return urlparse(url).netloc.lower()

    except Exception:
        return None


def percentile(values, q):

    if not values:
        return np.nan

    return float(np.percentile(values, q))


# =====================================================
# INSPECT ONE LANGUAGE
# =====================================================

def diagnose_language(language, config):

    repository = config["repository"]
    dataset_config = config["config"]

    print("\n" + "=" * 70)

    print("LANGUAGE:", language)
    print("REPOSITORY:", repository)
    print("CONFIGURATION:", dataset_config)

    print("=" * 70)

    dataset = load_dataset(
        repository,
        name=dataset_config,
        split="train",
        streaming=True
    )

    # Approximate randomized streaming sample.
    shuffled_dataset = dataset.shuffle(
        seed=RANDOM_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE
    )

    records = []

    for index, record in enumerate(
        shuffled_dataset
    ):

        text = record.get("text", "")

        if not isinstance(text, str):
            continue

        if not text.strip():
            continue

        char_count = len(text)

        letter_count = sum(
            character.isalpha()
            for character in text
        )

        letter_ratio = (
            letter_count / char_count
            if char_count > 0
            else 0
        )

        language_score = record.get(
            "language_score"
        )

        cluster_size = record.get(
            "minhash_cluster_size"
        )

        url = record.get("url")

        records.append({

            "language": language,

            "repository": repository,

            "configuration": dataset_config,

            "document_id": record.get("id"),

            "url": url,

            "domain": get_domain(url),

            "date": record.get("date"),

            "language_code": record.get("language"),

            "language_script": record.get(
                "language_script"
            ),

            "language_score": language_score,

            "document_characters": char_count,

            "whitespace_tokens": len(
                text.split()
            ),

            "alphabetic_character_ratio": letter_ratio,

            "minhash_cluster_size": cluster_size,

        })

        if len(records) >= SAMPLES_PER_LANGUAGE:
            break

    if not records:

        print("WARNING: No valid records collected.")
        return None, []

    df = pd.DataFrame(records)

    lengths = df[
        "document_characters"
    ].tolist()

    scores = pd.to_numeric(
        df["language_score"],
        errors="coerce"
    ).dropna().tolist()

    clusters = pd.to_numeric(
        df["minhash_cluster_size"],
        errors="coerce"
    ).dropna().tolist()

    # =================================================
    # QUALITY SUMMARY
    # =================================================

    summary = {

        "language": language,

        "repository": repository,

        "configuration": dataset_config,

        "sampled_documents": len(df),

        "unique_domains": df["domain"].nunique(),

        "documents_below_200_chars": int(
            (df["document_characters"] < 200).sum()
        ),

        "documents_below_500_chars": int(
            (df["document_characters"] < 500).sum()
        ),

        "documents_above_10000_chars": int(
            (df["document_characters"] > 10000).sum()
        ),

        "median_document_characters": float(
            np.median(lengths)
        ),

        "p10_document_characters": percentile(
            lengths, 10
        ),

        "p90_document_characters": percentile(
            lengths, 90
        ),

        "median_alphabetic_ratio": float(
            df["alphabetic_character_ratio"].median()
        ),

        "language_score_median": (
            float(np.median(scores))
            if scores else np.nan
        ),

        "language_score_p10": percentile(
            scores, 10
        ),

        "language_score_p90": percentile(
            scores, 90
        ),

        "median_minhash_cluster_size": (
            float(np.median(clusters))
            if clusters else np.nan
        ),

        "p95_minhash_cluster_size": percentile(
            clusters, 95
        ),

        "maximum_minhash_cluster_size": (
            max(clusters) if clusters else np.nan
        ),

    }

    # =================================================
    # DISPLAY
    # =================================================

    print("\nDOCUMENT STATISTICS")

    print("Documents inspected:", len(df))

    print("Unique domains:", summary["unique_domains"])

    print(
        "Median document length:",
        summary["median_document_characters"]
    )

    print(
        "10th percentile length:",
        summary["p10_document_characters"]
    )

    print(
        "90th percentile length:",
        summary["p90_document_characters"]
    )

    print(
        "Documents below 500 characters:",
        summary["documents_below_500_chars"]
    )

    print(
        "Documents above 10,000 characters:",
        summary["documents_above_10000_chars"]
    )

    print(
        "Median language score:",
        summary["language_score_median"]
    )

    print(
        "Median duplicate cluster size:",
        summary["median_minhash_cluster_size"]
    )

    print("\nSCRIPT DISTRIBUTION")

    print(
        df["language_script"]
        .value_counts(dropna=False)
        .to_dict()
    )

    print("\nTOP FIVE DOMAINS")

    print(
        df["domain"]
        .value_counts()
        .head(5)
        .to_dict()
    )

    print("\nDiagnostic completed for", language)

    return summary, records


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 70)

    print(
        "TCMB - RANDOMIZED CALIBRATION QUALITY DIAGNOSTICS"
    )

    print("=" * 70)

    print("Sample size:", SAMPLES_PER_LANGUAGE)

    print("Shuffle buffer:", SHUFFLE_BUFFER_SIZE)

    print("Random seed:", RANDOM_SEED)

    all_summaries = []

    all_records = []

    for language, config in SOURCES.items():

        try:

            summary, records = diagnose_language(
                language,
                config
            )

            if summary is not None:

                all_summaries.append(summary)

                all_records.extend(records)

        except Exception as error:

            print(
                f"\nERROR for {language}: "
                f"{type(error).__name__}: {error}"
            )

    # =================================================
    # SAVE RESULTS
    # =================================================

    if not all_summaries:

        raise RuntimeError(
            "No language samples were successfully inspected."
        )

    summary_df = pd.DataFrame(
        all_summaries
    )

    records_df = pd.DataFrame(
        all_records
    )

    summary_path = (
        RESULTS_DIR
        / "calibration_quality_summary.csv"
    )

    records_path = (
        RESULTS_DIR
        / "calibration_quality_sample_metadata.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    records_df.to_csv(
        records_path,
        index=False
    )

    # =================================================
    # FINAL REPORT
    # =================================================

    print("\n" + "=" * 70)

    print(
        "CALIBRATION QUALITY DIAGNOSTICS COMPLETED"
    )

    print("=" * 70)

    print("\nSummary:")

    print(
        summary_df[
            [
                "language",
                "sampled_documents",
                "unique_domains",
                "median_document_characters",
                "documents_below_500_chars",
                "language_score_median",
                "median_minhash_cluster_size",
            ]
        ].to_string(index=False)
    )

    print("\nFiles saved:")

    print(summary_path)

    print(records_path)

    print(
        "\nNote: No document text was saved."
    )


if __name__ == "__main__":

    main()