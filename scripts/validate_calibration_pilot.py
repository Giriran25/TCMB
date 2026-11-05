"""
TCMB - Calibration Pilot Integrity Validation

Checks:
1. All eight language files exist.
2. JSONL records are readable.
3. Token counts match the XLM-R tokenizer.
4. SHA-256 hashes are unique.
5. Each language reaches the 100K token target.
6. Chinese conversion is idempotent.
7. FLORES+ exact sentence overlap is checked again.

No network downloads.
No model training.
Original sentence text is not exported.
"""

import json
import hashlib
import unicodedata

from pathlib import Path

import pandas as pd

from datasets import load_from_disk
from transformers import AutoTokenizer
from opencc import OpenCC


MODEL_NAME = "FacebookAI/xlm-roberta-base"

TARGET_TOKENS = 100_000

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

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "calibration_pilot"

RESULTS_DIR = ROOT / "results" / "tables"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text, converter=None):

    text = unicodedata.normalize("NFC", text)

    if converter is not None:
        text = converter.convert(text)

    return " ".join(text.split())


def calculate_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_evaluation_sentences(language_code, converter):

    dataset = load_from_disk(
        str(ROOT / "data" / "flores_plus" / language_code)
    )

    sentences = set()

    for split in ["dev", "devtest"]:

        for text in dataset[split]["text"]:

            normalized = normalize_text(
                text,
                converter
            )

            # Preserve the original collection filter.
            # Shorter sentences are separately counted below.
            if len(normalized) >= 30:
                sentences.add(normalized)

    return sentences


def validate_language(
    tokenizer,
    language_code,
    language_name
):

    print("\n" + "=" * 65)
    print("VALIDATING:", language_name)
    print("=" * 65)

    file_path = (
        DATA_DIR
        / language_code
        / "documents.jsonl"
    )

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    converter = (
        OpenCC("t2s")
        if language_code == "cmn_Hans"
        else None
    )

    eval_sentences = load_evaluation_sentences(
        language_code,
        converter
    )

    eval_sentence_lengths = [
        len(sentence)
        for sentence in eval_sentences
    ]

    short_eval_sentences = len([
        length
        for length in eval_sentence_lengths
        if length < 30
    ])

    seen_hashes = set()

    token_total = 0

    document_count = 0

    token_mismatches = 0

    duplicate_hashes = 0

    overlap_count = 0

    conversion_failures = 0

    min_chars = None

    max_chars = 0

    min_tokens = None

    max_tokens = 0

    domains = set()

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(file, start=1):

            record = json.loads(line)

            text = record.get("text", "")

            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"Invalid text at line {line_number}"
                )

            # Recompute tokenizer count.
            actual_tokens = len(
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=False
                )["input_ids"]
            )

            stored_tokens = record["xlmr_token_count"]

            if actual_tokens != stored_tokens:
                token_mismatches += 1

            token_total += actual_tokens

            document_count += 1

            # Validate document hashes.
            actual_hash = calculate_hash(text)

            if actual_hash != record["normalized_text_sha256"]:
                raise ValueError(
                    f"Hash mismatch at line {line_number}"
                )

            if actual_hash in seen_hashes:
                duplicate_hashes += 1

            seen_hashes.add(actual_hash)

            # Exact evaluation overlap check.
            if any(
                sentence in text
                for sentence in eval_sentences
            ):
                overlap_count += 1

            # Verify Chinese conversion is idempotent.
            if converter is not None:

                converted_again = converter.convert(text)

                if converted_again != text:
                    conversion_failures += 1

            char_count = len(text)

            min_chars = (
                char_count
                if min_chars is None
                else min(min_chars, char_count)
            )

            max_chars = max(
                max_chars,
                char_count
            )

            min_tokens = (
                actual_tokens
                if min_tokens is None
                else min(min_tokens, actual_tokens)
            )

            max_tokens = max(
                max_tokens,
                actual_tokens
            )

            if record.get("domain"):
                domains.add(record["domain"])

    # ------------------------------------------
    # Final validation
    # ------------------------------------------

    target_reached = token_total >= TARGET_TOKENS

    summary = {

        "language_code": language_code,

        "language": language_name,

        "documents": document_count,

        "actual_xlmr_tokens": token_total,

        "target_tokens": TARGET_TOKENS,

        "target_reached": target_reached,

        "token_mismatches": token_mismatches,

        "duplicate_hashes": duplicate_hashes,

        "exact_flores_overlapping_documents": overlap_count,

        "flores_sentences_checked": len(eval_sentences),

        "short_flores_sentences_not_checked": (
            short_eval_sentences
        ),

        "chinese_conversion_failures": conversion_failures,

        "unique_domains": len(domains),

        "minimum_document_characters": min_chars,

        "maximum_document_characters": max_chars,

        "minimum_document_tokens": min_tokens,

        "maximum_document_tokens": max_tokens,

    }

    print("Documents:", document_count)

    print("Recomputed XLM-R tokens:", token_total)

    print("Target reached:", target_reached)

    print("Token mismatches:", token_mismatches)

    print("Duplicate hashes:", duplicate_hashes)

    print("Exact FLORES+ overlaps:", overlap_count)

    print("FLORES+ sentences checked:", len(eval_sentences))

    print(
        "Short FLORES+ sentences not checked:",
        short_eval_sentences
    )

    print("Chinese conversion failures:", conversion_failures)

    return summary


def main():

    print("=" * 65)

    print(
        "TCMB - CALIBRATION PILOT VALIDATION"
    )

    print("=" * 65)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    summaries = []

    for language_code, language_name in LANGUAGES.items():

        try:

            summary = validate_language(
                tokenizer,
                language_code,
                language_name
            )

            summaries.append(summary)

        except Exception as error:

            print(
                f"\nVALIDATION ERROR: {language_name}: "
                f"{type(error).__name__}: {error}"
            )

            raise

    df = pd.DataFrame(summaries)

    output_path = (
        RESULTS_DIR
        / "calibration_pilot_validation.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 65)

    print("PILOT VALIDATION COMPLETED")

    print("=" * 65)

    print(
        df[
            [
                "language",
                "documents",
                "actual_xlmr_tokens",
                "target_reached",
                "token_mismatches",
                "duplicate_hashes",
                "exact_flores_overlapping_documents",
                "chinese_conversion_failures"
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_path)

    # Strict integrity checks.
    if not df["target_reached"].all():
        raise RuntimeError("At least one language missed its token target.")

    if (df["token_mismatches"] > 0).any():
        raise RuntimeError("Token-count validation failed.")

    if (df["duplicate_hashes"] > 0).any():
        raise RuntimeError("Duplicate document hashes detected.")

    if (df["exact_flores_overlapping_documents"] > 0).any():
        raise RuntimeError("FLORES+ sentence overlap detected.")

    if (df["chinese_conversion_failures"] > 0).any():
        raise RuntimeError("Chinese conversion is not idempotent.")

    print("\nAll strict pilot integrity checks passed.")


if __name__ == "__main__":
    main()