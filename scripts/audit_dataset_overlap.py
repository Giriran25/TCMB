"""
TCMB - SIB-200 / FLORES+ Dataset Overlap Audit

Purpose:
    Identify sentence-level overlap between SIB-200 and FLORES+.

Languages:
    English, Tamil, Telugu, Kannada, Malayalam,
    Hindi, Chinese and Japanese.

SIB-200 splits:
    train, validation, test

FLORES+ splits:
    dev, devtest

Normalization:
    1. Unicode NFKC normalization
    2. Case folding
    3. Whitespace normalization

Privacy:
    Original sentence text is never written to CSV.
    Only identifiers and SHA-256 hashes are exported.

Outputs:
    results/tables/dataset_overlap_summary.csv
    results/tables/dataset_overlap_details.csv
"""

import hashlib
import unicodedata

from collections import defaultdict
from pathlib import Path

import pandas as pd

from datasets import load_from_disk


# =========================================================
# CONFIGURATION
# =========================================================

LANGUAGES = {
    "eng_Latn": {
        "name": "English",
        "sib_code": "eng_Latn",
    },
    "tam_Taml": {
        "name": "Tamil",
        "sib_code": "tam_Taml",
    },
    "tel_Telu": {
        "name": "Telugu",
        "sib_code": "tel_Telu",
    },
    "kan_Knda": {
        "name": "Kannada",
        "sib_code": "kan_Knda",
    },
    "mal_Mlym": {
        "name": "Malayalam",
        "sib_code": "mal_Mlym",
    },
    "hin_Deva": {
        "name": "Hindi",
        "sib_code": "hin_Deva",
    },
    "cmn_Hans": {
        "name": "Chinese",
        "sib_code": "zho_Hans",
    },
    "jpn_Jpan": {
        "name": "Japanese",
        "sib_code": "jpn_Jpan",
    },
}


SIB_SPLITS = [
    "train",
    "validation",
    "test",
]

FLORES_SPLITS = [
    "dev",
    "devtest",
]


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SIB_DIRECTORY = PROJECT_ROOT / "data" / "sib200"

FLORES_DIRECTORY = PROJECT_ROOT / "data" / "flores_plus"

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "tables"
)

RESULTS_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text: str) -> str:
    """
    Apply conservative normalization before comparison.

    Preserves punctuation and lexical content.
    """

    if not isinstance(text, str):
        raise TypeError(
            "Dataset text must be a string."
        )

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.casefold()

    text = " ".join(text.split())

    return text


def generate_text_hash(normalized_text: str) -> str:
    """
    Generate SHA-256 fingerprint.

    Original text is not stored in output files.
    """

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


# =========================================================
# BUILD DATASET INDEX
# =========================================================

def build_index(
    dataset,
    identifier_column: str,
    source_name: str,
    split_name: str,
):
    """
    Build an index:

        normalized_text -> list of record identifiers

    Duplicate sentences are retained.
    """

    if "text" not in dataset.column_names:
        raise ValueError(
            f"'text' column missing in {source_name}/{split_name}"
        )

    if identifier_column not in dataset.column_names:
        raise ValueError(
            f"'{identifier_column}' missing in "
            f"{source_name}/{split_name}"
        )

    index = defaultdict(list)

    for record in dataset:

        original_text = record["text"]

        if not isinstance(original_text, str):
            continue

        normalized = normalize_text(
            original_text
        )

        if not normalized:
            continue

        identifier = str(
            record[identifier_column]
        )

        index[normalized].append(identifier)

    return index


# =========================================================
# LOAD SIB-200
# =========================================================

def load_sib200(language_code: str):

    dataset_path = (
        SIB_DIRECTORY
        / language_code
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"SIB-200 dataset not found: {dataset_path}"
        )

    return load_from_disk(
        str(dataset_path)
    )


# =========================================================
# LOAD FLORES+
# =========================================================

def load_flores(language_code: str):

    dataset_path = (
        FLORES_DIRECTORY
        / language_code
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"FLORES+ dataset not found: {dataset_path}"
        )

    return load_from_disk(
        str(dataset_path)
    )


# =========================================================
# AUDIT ONE SPLIT PAIR
# =========================================================

def compare_split_pair(
    language_code,
    language_name,
    sib_split,
    flores_split,
    sib_index,
    flores_index,
):
    """
    Compare one SIB-200 split against one FLORES+ split.
    """

    common_texts = (
        set(sib_index.keys())
        &
        set(flores_index.keys())
    )

    sib_matching_rows = sum(
        len(sib_index[text])
        for text in common_texts
    )

    flores_matching_rows = sum(
        len(flores_index[text])
        for text in common_texts
    )

    matched_pairs = sum(
        len(sib_index[text])
        * len(flores_index[text])
        for text in common_texts
    )

    sib_total = sum(
        len(records)
        for records in sib_index.values()
    )

    flores_total = sum(
        len(records)
        for records in flores_index.values()
    )

    summary = {
        "language_code": language_code,
        "language": language_name,

        "sib_split": sib_split,
        "flores_split": flores_split,

        "sib_total_rows": sib_total,
        "flores_total_rows": flores_total,

        "unique_matching_sentences": len(common_texts),

        "sib_matching_rows": sib_matching_rows,

        "flores_matching_rows": flores_matching_rows,

        "matched_record_pairs": matched_pairs,

        "sib_overlap_percentage": (
            100 * sib_matching_rows / sib_total
            if sib_total else 0
        ),

        "flores_overlap_percentage": (
            100 * flores_matching_rows / flores_total
            if flores_total else 0
        ),
    }

    details = []

    for normalized in sorted(common_texts):

        text_hash = generate_text_hash(
            normalized
        )

        for sib_id in sib_index[normalized]:

            for flores_id in flores_index[normalized]:

                details.append({
                    "language_code": language_code,
                    "language": language_name,

                    "sib_split": sib_split,
                    "flores_split": flores_split,

                    "sib_index_id": sib_id,
                    "flores_id": flores_id,

                    "normalized_text_sha256": text_hash,

                    "match_type": "normalized_exact",
                })

    return summary, details


# =========================================================
# MAIN AUDIT
# =========================================================

def main():

    print("=" * 75)

    print(
        "TCMB - SIB-200 / FLORES+ DATASET OVERLAP AUDIT"
    )

    print("=" * 75)

    all_summaries = []

    all_details = []

    for language_code, config in LANGUAGES.items():

        language_name = config["name"]

        sib_code = config["sib_code"]

        print("\n" + "-" * 75)

        print(
            f"Language: {language_name} ({language_code})"
        )

        print("-" * 75)

        # -------------------------------------------------
        # Load datasets
        # -------------------------------------------------

        sib_dataset = load_sib200(sib_code)

        flores_dataset = load_flores(
            language_code
        )

        print(
            "SIB-200 splits:",
            list(sib_dataset.keys())
        )

        print(
            "FLORES+ splits:",
            list(flores_dataset.keys())
        )

        # -------------------------------------------------
        # Build FLORES+ indices
        # -------------------------------------------------

        flores_indices = {}

        for flores_split in FLORES_SPLITS:

            flores_records = flores_dataset[
                flores_split
            ]

            flores_indices[flores_split] = build_index(
                dataset=flores_records,
                identifier_column="id",
                source_name="FLORES+",
                split_name=flores_split,
            )

        # Union of both FLORES+ splits
        flores_union = defaultdict(list)

        for split_name, index in flores_indices.items():

            for normalized, identifiers in index.items():

                for identifier in identifiers:

                    flores_union[normalized].append({
                        "split": split_name,
                        "id": identifier,
                    })

        # -------------------------------------------------
        # Compare every SIB split
        # -------------------------------------------------

        for sib_split in SIB_SPLITS:

            sib_records = sib_dataset[sib_split]

            sib_index = build_index(
                dataset=sib_records,
                identifier_column="index_id",
                source_name="SIB-200",
                split_name=sib_split,
            )

            # Compare against FLORES dev and devtest
            for flores_split in FLORES_SPLITS:

                summary, details = compare_split_pair(
                    language_code=language_code,
                    language_name=language_name,
                    sib_split=sib_split,
                    flores_split=flores_split,
                    sib_index=sib_index,
                    flores_index=flores_indices[flores_split],
                )

                all_summaries.append(summary)

                all_details.extend(details)

            # -------------------------------------------------
            # Compare against the combined FLORES+ splits
            # -------------------------------------------------

            combined_flores_index = {}

            for normalized, records in flores_union.items():

                combined_flores_index[normalized] = [
                    str(record["id"])
                    for record in records
                ]

            summary, details = compare_split_pair(
                language_code=language_code,
                language_name=language_name,
                sib_split=sib_split,
                flores_split="dev+devtest",
                sib_index=sib_index,
                flores_index=combined_flores_index,
            )

            all_summaries.append(summary)

            # For the combined comparison, retain split
            # provenance in the detailed report.
            for detail in details:

                normalized = next(
                    (
                        text
                        for text in sib_index
                        if generate_text_hash(text)
                        == detail["normalized_text_sha256"]
                    ),
                    None,
                )

                if normalized is not None:

                    matching_flores_records = flores_union[
                        normalized
                    ]

                    for flores_record in matching_flores_records:

                        all_details.append({
                            "language_code": language_code,
                            "language": language_name,

                            "sib_split": sib_split,

                            "flores_split": flores_record["split"],

                            "sib_index_id": detail["sib_index_id"],

                            "flores_id": flores_record["id"],

                            "normalized_text_sha256": (
                                detail["normalized_text_sha256"]
                            ),

                            "match_type": "normalized_exact",
                        })

            # -------------------------------------------------
            # Print split summary
            # -------------------------------------------------

            combined_overlap = summary[
                "sib_overlap_percentage"
            ]

            print(
                f"\nSIB {sib_split}: "
                f"{len(sib_records)} rows"
            )

            print(
                "Matching FLORES+ dev/devtest rows:",
                summary["sib_matching_rows"],
            )

            print(
                "Unique matching sentences:",
                summary["unique_matching_sentences"],
            )

            print(
                "SIB overlap percentage:",
                f"{combined_overlap:.2f}%",
            )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    summary_df = pd.DataFrame(
        all_summaries
    )

    details_df = pd.DataFrame(
        all_details
    )

    summary_path = (
        RESULTS_DIRECTORY
        / "dataset_overlap_summary.csv"
    )

    details_path = (
        RESULTS_DIRECTORY
        / "dataset_overlap_details.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    details_df.to_csv(
        details_path,
        index=False,
    )

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print("\n" + "=" * 75)

    print(
        "DATASET OVERLAP AUDIT COMPLETED"
    )

    print("=" * 75)

    print("\nSummary report:")

    print(summary_path)

    print("\nDetailed report:")

    print(details_path)

    print("\nTotal comparison records:")

    print(len(summary_df))

    print("\nTotal matching record pairs:")

    print(len(details_df))

    print(
        "\nOriginal sentence text was not exported."
    )

    print(
        "\nIMPORTANT: Any overlap must be handled "
        "before training or final evaluation."
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()