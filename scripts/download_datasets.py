from pathlib import Path
from datasets import load_dataset
import shutil

DATASETS = {
    "flores_plus": {
        "repository": "openlanguagedata/flores_plus",
        "languages": [
            "eng_Latn",
            "tam_Taml",
            "tel_Telu",
            "kan_Knda",
            "mal_Mlym",
            "hin_Deva",
            "cmn_Hans",
            "jpn_Jpan"
        ]
    },

    "sib200": {
        "repository": "Davlan/sib200",
        "languages": [
            "eng_Latn",
            "tam_Taml",
            "tel_Telu",
            "kan_Knda",
            "mal_Mlym",
            "hin_Deva",
            "zho_Hans",
            "jpn_Jpan"
        ]
    }
}

BASE_DIR = Path("data")


def download_dataset(dataset_name, repository, language):

    output_dir = BASE_DIR / dataset_name / language
    manifest = output_dir / "dataset_dict.json"

    if manifest.exists():
        print(f"[EXISTS] {dataset_name}/{language}")
        return True

    print(f"\nDownloading {dataset_name} - {language}")

    temp_dir = output_dir.with_name(output_dir.name + ".partial")

    try:
        dataset = load_dataset(
            repository,
            name=language
        )

        output_dir.parent.mkdir(parents=True, exist_ok=True)

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        dataset.save_to_disk(str(temp_dir))

        if output_dir.exists():
            raise FileExistsError(
                f"Incomplete output directory: {output_dir}"
            )

        temp_dir.rename(output_dir)

        print(f"[SAVED] {output_dir}")

        for split, data in dataset.items():
            print(f"  {split}: {len(data)} examples")

        return True

    except Exception as error:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        print(f"[FAILED] {dataset_name}/{language}")
        print(f"Reason: {error}")
        return False


def main():

    failures = []

    for dataset_name, config in DATASETS.items():

        print(f"\n{'=' * 55}")
        print(f"DATASET: {dataset_name.upper()}")
        print(f"{'=' * 55}")

        for language in config["languages"]:

            success = download_dataset(
                dataset_name,
                config["repository"],
                language
            )

            if not success:
                failures.append((dataset_name, language))

    print("\n========== DOWNLOAD SUMMARY ==========")

    if failures:
        print("Some downloads failed:")

        for dataset, language in failures:
            print(f"- {dataset}: {language}")
    else:
        print("All 16 dataset configurations downloaded successfully!")


if __name__ == "__main__":
    main()