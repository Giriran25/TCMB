"""
TCMB - Calibration Corpus Sample Inspection

Inspects a small number of records from FineWeb
and FineWeb-2 using streaming.

No complete dataset download.
No training.
No files are modified.
"""

from itertools import islice

from datasets import load_dataset


# -------------------------------------------------
# DATASET CONFIGURATION
# -------------------------------------------------

SOURCES = {
    "English": {
        "repository": "HuggingFaceFW/fineweb",
        "config": "default",
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


SAMPLES_PER_LANGUAGE = 5


# -------------------------------------------------
# INSPECT ONE LANGUAGE
# -------------------------------------------------

def inspect_language(language, config):

    repository = config["repository"]
    dataset_config = config["config"]

    print("\n" + "=" * 75)
    print(f"LANGUAGE: {language}")
    print("=" * 75)

    print("Repository:", repository)
    print("Configuration:", dataset_config)
    print("Split: train")
    print("Streaming: enabled")

    try:

        dataset = load_dataset(
            repository,
            name=dataset_config,
            split="train",
            streaming=True,
        )

        samples = iter(
            islice(dataset, SAMPLES_PER_LANGUAGE)
        )

        count = 0

        for index, record in enumerate(samples, start=1):

            text = record.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            count += 1

            print(f"\nSample {index}")

            print("ID:", record.get("id", "Unavailable"))

            print(
                "Language:",
                record.get("language", "Unavailable"),
            )

            print(
                "Script:",
                record.get("language_script", "Not provided"),
            )

            print(
                "Language confidence:",
                record.get("language_score", "Not provided"),
            )

            print(
                "Date:",
                record.get("date", "Not provided"),
            )

            print(
                "Document length:",
                len(text),
                "characters",
            )

            print(
                "MinHash cluster size:",
                record.get("minhash_cluster_size", "Not provided"),
            )

            print(
                "Text preview:",
                text[:250].replace("\n", " "),
            )

        print(f"\nValid samples inspected: {count}")

        if count == 0:
            print("WARNING: No valid text records found.")

    except Exception as error:

        print("\nERROR while inspecting this language:")
        print(type(error).__name__, error)


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    print("=" * 75)
    print("TCMB - CALIBRATION CORPUS SAMPLE INSPECTION")
    print("=" * 75)

    print("\nThis is a streaming inspection.")
    print("Only a maximum of five records per language will be read.")

    for language, config in SOURCES.items():

        inspect_language(
            language,
            config,
        )

    print("\n" + "=" * 75)
    print("CALIBRATION SAMPLE INSPECTION COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()