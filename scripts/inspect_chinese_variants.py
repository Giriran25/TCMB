"""
TCMB - Chinese Script Variant Inspection

Inspects FineWeb-2 cmn_Hani documents and estimates
how frequently Traditional-to-Simplified conversion
changes the original text.

This is a diagnostic, not a perfect language-variety classifier.
"""

from itertools import islice

from datasets import load_dataset
from opencc import OpenCC


DATASET = "HuggingFaceFW/fineweb-2"
CONFIG = "cmn_Hani"

SAMPLE_SIZE = 200
BUFFER_SIZE = 10000
SEED = 42


def main():

    print("=" * 65)
    print("TCMB - CHINESE VARIANT INSPECTION")
    print("=" * 65)

    converter = OpenCC("t2s")

    print("\nLoading Chinese calibration stream...")

    dataset = load_dataset(
        DATASET,
        name=CONFIG,
        split="train",
        streaming=True
    )

    dataset = dataset.shuffle(
        seed=SEED,
        buffer_size=BUFFER_SIZE
    )

    converted_count = 0
    unchanged_count = 0
    total_characters = 0
    changed_characters = 0

    for index, record in enumerate(
        islice(dataset, SAMPLE_SIZE),
        start=1
    ):

        text = record.get("text", "")

        if not isinstance(text, str) or not text.strip():
            continue

        simplified = converter.convert(text)

        changed = text != simplified

        if changed:
            converted_count += 1
        else:
            unchanged_count += 1

        total_characters += len(text)

        changed_characters += sum(
            1 for original, converted in zip(text, simplified)
            if original != converted
        )

        if index <= 5:

            print(f"\nSample {index}")

            print("Original:")
            print(text[:200])

            print("\nSimplified conversion:")
            print(simplified[:200])

            print(
                "\nConversion changed text:",
                changed
            )

    inspected = converted_count + unchanged_count

    print("\n" + "=" * 65)
    print("CHINESE VARIANT SUMMARY")
    print("=" * 65)

    print("Documents inspected:", inspected)
    print("Documents changed:", converted_count)
    print("Documents unchanged:", unchanged_count)

    print(
        "Documents requiring conversion (%):",
        round(100 * converted_count / inspected, 2)
        if inspected else 0
    )

    print("Original characters:", total_characters)
    print("Changed aligned character positions:", changed_characters)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()