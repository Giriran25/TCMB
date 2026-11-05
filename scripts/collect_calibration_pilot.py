"""
TCMB - Calibration Corpus Pilot Collection

Collects a balanced 100K XLM-R-token pilot
for each of eight languages.

Sources:
    English: FineWeb sample-10BT
    Others: FineWeb-2

Features:
    - Streaming collection
    - Fixed random seed
    - Script validation
    - Document quality filtering
    - Chinese Traditional-to-Simplified conversion
    - Exact duplicate filtering
    - FLORES+ evaluation sentence overlap filtering
    - XLM-R token-budget tracking

Output:
    data/calibration_pilot/<language_code>/documents.jsonl
    results/tables/calibration_pilot_summary.csv

Raw corpus data is kept locally and should not be
committed to the public GitHub repository.
"""

import hashlib
import json
import unicodedata

from pathlib import Path
from urllib.parse import urlparse

import regex
import pandas as pd

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
from opencc import OpenCC


# =====================================================
# CONFIGURATION
# =====================================================

MODEL_NAME = "FacebookAI/xlm-roberta-base"

TARGET_TOKENS = 100_000

MIN_DOCUMENT_CHARS = 300
MIN_DOCUMENT_TOKENS = 100
MAX_DOCUMENT_TOKENS = 10_000

MIN_SCRIPT_RATIO = 0.55

MAX_SCANNED_DOCUMENTS = 50_000

SHUFFLE_BUFFER_SIZE = 10_000

RANDOM_SEED = 42


# =====================================================
# LANGUAGE SOURCES
# =====================================================

LANGUAGES = {

    "eng_Latn": {
        "name": "English",
        "repository": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "language": "en",
        "script": None,
        "script_pattern": r"\p{Script=Latin}",
        "chinese": False,
    },

    "tam_Taml": {
        "name": "Tamil",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "tam_Taml",
        "language": "tam",
        "script": "Taml",
        "script_pattern": r"\p{Script=Tamil}",
        "chinese": False,
    },

    "tel_Telu": {
        "name": "Telugu",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "tel_Telu",
        "language": "tel",
        "script": "Telu",
        "script_pattern": r"\p{Script=Telugu}",
        "chinese": False,
    },

    "kan_Knda": {
        "name": "Kannada",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "kan_Knda",
        "language": "kan",
        "script": "Knda",
        "script_pattern": r"\p{Script=Kannada}",
        "chinese": False,
    },

    "mal_Mlym": {
        "name": "Malayalam",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "mal_Mlym",
        "language": "mal",
        "script": "Mlym",
        "script_pattern": r"\p{Script=Malayalam}",
        "chinese": False,
    },

    "hin_Deva": {
        "name": "Hindi",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "hin_Deva",
        "language": "hin",
        "script": "Deva",
        "script_pattern": r"\p{Script=Devanagari}",
        "chinese": False,
    },

    "cmn_Hans": {
        "name": "Chinese",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "cmn_Hani",
        "language": "cmn",
        "script": "Hani",
        "script_pattern": r"\p{Script=Han}",
        "chinese": True,
    },

    "jpn_Jpan": {
        "name": "Japanese",
        "repository": "HuggingFaceFW/fineweb-2",
        "config": "jpn_Jpan",
        "language": "jpn",
        "script": "Jpan",
        "script_pattern": (
            r"(?:\p{Script=Han}|"
            r"\p{Script=Hiragana}|"
            r"\p{Script=Katakana})"
        ),
        "chinese": False,
    },
}


# =====================================================
# PATHS
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "calibration_pilot"

RESULTS_DIR = ROOT / "results" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text, converter=None):

    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)

    # Convert all Chinese documents consistently.
    if converter is not None:
        text = converter.convert(text)

    # Normalize whitespace without removing punctuation.
    text = regex.sub(r"\s+", " ", text).strip()

    return text


def calculate_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# =====================================================
# SCRIPT VALIDATION
# =====================================================

def calculate_script_ratio(text, pattern):

    letters = regex.findall(r"\p{L}", text)

    if not letters:
        return 0.0

    script_regex = regex.compile(pattern)

    matching_letters = sum(
        1 for character in letters
        if script_regex.fullmatch(character)
    )

    return matching_letters / len(letters)


# =====================================================
# DOMAIN EXTRACTION
# =====================================================

def extract_domain(url):

    if not isinstance(url, str):
        return ""

    try:
        return urlparse(url).netloc.lower()

    except Exception:
        return ""


# =====================================================
# FLORES+ OVERLAP INDEX
# =====================================================

def load_evaluation_sentences(language_code, converter):

    print(
        f"Loading FLORES+ exclusion index: {language_code}"
    )

    path = ROOT / "data" / "flores_plus" / language_code

    dataset = load_from_disk(str(path))

    evaluation_sentences = []

    for split in ["dev", "devtest"]:

        for text in dataset[split]["text"]:

            normalized = normalize_text(
                text,
                converter
            )

            if len(normalized) >= 30:
                evaluation_sentences.append(normalized)

    return evaluation_sentences


def contains_evaluation_sentence(text, evaluation_sentences):

    for sentence in evaluation_sentences:

        if sentence in text:
            return True

    return False


# =====================================================
# COLLECT ONE LANGUAGE
# =====================================================

def collect_language(
    tokenizer,
    language_code,
    config
):

    language_name = config["name"]

    repository = config["repository"]

    dataset_config = config["config"]

    print("\n" + "=" * 70)
    print("COLLECTING:", language_name)
    print("=" * 70)

    converter = (
        OpenCC("t2s")
        if config["chinese"]
        else None
    )

    # ---------------------------------------------
    # Load evaluation exclusion sentences
    # ---------------------------------------------

    evaluation_sentences = load_evaluation_sentences(
        language_code,
        converter
    )

    evaluation_sentences = list(
        set(evaluation_sentences)
    )

    print(
        "FLORES+ exclusion sentences:",
        len(evaluation_sentences)
    )

    # ---------------------------------------------
    # Load dataset as a stream
    # ---------------------------------------------

    dataset = load_dataset(
        repository,
        name=dataset_config,
        split="train",
        streaming=True
    )

    dataset = dataset.shuffle(
        seed=RANDOM_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE
    )

    # ---------------------------------------------
    # Initialize counters
    # ---------------------------------------------

    accepted_documents = 0
    scanned_documents = 0

    token_total = 0

    short_documents = 0
    wrong_language = 0
    wrong_script = 0
    low_script_ratio = 0
    duplicate_documents = 0
    evaluation_overlaps = 0
    invalid_documents = 0
    oversized_documents = 0

    exact_hashes = set()
    domains = set()

    output_path = (
        OUTPUT_DIR
        / language_code
        / "documents.jsonl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------
    # Stream and filter documents
    # ---------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        for record in dataset:

            scanned_documents += 1

            if scanned_documents > MAX_SCANNED_DOCUMENTS:
                break

            text = record.get("text", "")

            if not isinstance(text, str) or not text.strip():

                invalid_documents += 1
                continue

            # Language metadata validation.
            if record.get("language") != config["language"]:

                wrong_language += 1
                continue

            # Script metadata validation.
            expected_script = config["script"]

            actual_script = record.get("language_script")

            if (
                expected_script is not None
                and actual_script != expected_script
            ):

                wrong_script += 1
                continue

            # Normalize and optionally convert.
            normalized = normalize_text(
                text,
                converter
            )

            if len(normalized) < MIN_DOCUMENT_CHARS:

                short_documents += 1
                continue

            # Script purity check.
            script_ratio = calculate_script_ratio(
                normalized,
                config["script_pattern"]
            )

            if script_ratio < MIN_SCRIPT_RATIO:

                low_script_ratio += 1
                continue

            # Exact full-document deduplication.
            text_hash = calculate_hash(normalized)

            if text_hash in exact_hashes:

                duplicate_documents += 1
                continue

            # Exclude documents containing exact evaluation sentences.
            if contains_evaluation_sentence(
                normalized,
                evaluation_sentences
            ):

                evaluation_overlaps += 1
                continue

            # XLM-R token count.
            input_ids = tokenizer(
                normalized,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            token_count = len(input_ids)

            if token_count < MIN_DOCUMENT_TOKENS:

                short_documents += 1
                continue

            if token_count > MAX_DOCUMENT_TOKENS:

                oversized_documents += 1
                continue

            # Store accepted document.
            metadata = {

                "language_code": language_code,

                "language": language_name,

                "source": repository,

                "configuration": dataset_config,

                "document_id": record.get("id"),

                "url": record.get("url"),

                "domain": extract_domain(
                    record.get("url")
                ),

                "date": record.get("date"),

                "dump": record.get("dump"),

                "language_score": record.get(
                    "language_score"
                ),

                "minhash_cluster_size": record.get(
                    "minhash_cluster_size"
                ),

                "original_characters": len(text),

                "normalized_characters": len(normalized),

                "xlmr_token_count": token_count,

                "script_ratio": script_ratio,

                "normalized_text_sha256": text_hash,

                "chinese_conversion": (
                    "OpenCC_t2s"
                    if config["chinese"]
                    else "not_applicable"
                ),

                "text": normalized,
            }

            output_file.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                ) + "\n"
            )

            exact_hashes.add(text_hash)

            domain = metadata["domain"]

            if domain:
                domains.add(domain)

            accepted_documents += 1

            token_total += token_count

            # Progress update.
            if accepted_documents % 25 == 0:

                print(
                    f"Accepted: {accepted_documents} | "
                    f"Tokens: {token_total:,}"
                )

            # Stop when target is reached.
            if token_total >= TARGET_TOKENS:
                break

    # ---------------------------------------------
    # Language summary
    # ---------------------------------------------

    target_reached = token_total >= TARGET_TOKENS

    summary = {

        "language_code": language_code,

        "language": language_name,

        "source": repository,

        "configuration": dataset_config,

        "target_tokens": TARGET_TOKENS,

        "actual_tokens": token_total,

        "target_reached": target_reached,

        "accepted_documents": accepted_documents,

        "scanned_documents": scanned_documents,

        "unique_domains": len(domains),

        "short_documents_rejected": short_documents,

        "wrong_language_rejected": wrong_language,

        "wrong_script_rejected": wrong_script,

        "low_script_ratio_rejected": low_script_ratio,

        "duplicate_documents_rejected": duplicate_documents,

        "evaluation_overlaps_rejected": evaluation_overlaps,

        "oversized_documents_rejected": oversized_documents,

        "invalid_documents_rejected": invalid_documents,

        "output_file": str(output_path),
    }

    print("\nCollection summary:")

    for key, value in summary.items():

        print(f"{key}: {value}")

    return summary


# =====================================================
# MAIN
# =====================================================

def main():

    print("=" * 70)

    print(
        "TCMB - 100K TOKEN CALIBRATION PILOT"
    )

    print("=" * 70)

    print("Model:", MODEL_NAME)

    print("Target per language:", TARGET_TOKENS)

    print("Total target:", TARGET_TOKENS * len(LANGUAGES))

    print("Random seed:", RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    summaries = []

    for language_code, config in LANGUAGES.items():

        try:

            summary = collect_language(
                tokenizer,
                language_code,
                config
            )

            summaries.append(summary)

        except Exception as error:

            print(
                f"\nERROR: {language_code}: "
                f"{type(error).__name__}: {error}"
            )

            summaries.append({

                "language_code": language_code,

                "language": config["name"],

                "target_reached": False,

                "error": str(error),
            })

    # ---------------------------------------------
    # Save summary
    # ---------------------------------------------

    summary_df = pd.DataFrame(summaries)

    summary_path = (
        RESULTS_DIR
        / "calibration_pilot_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 70)

    print("CALIBRATION PILOT COMPLETED")

    print("=" * 70)

    print(
        summary_df.to_string(index=False)
    )

    print("\nSummary saved:")
    print(summary_path)

    print(
        "\nCheck target_reached for every language "
        "before proceeding."
    )


if __name__ == "__main__":
    main()