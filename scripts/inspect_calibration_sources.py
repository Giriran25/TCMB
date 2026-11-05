"""
TCMB - Calibration Corpus Availability Check

Checks FineWeb-2 and FineWeb configurations.
Does not download dataset contents.
"""

from datasets import get_dataset_config_names


FINEWEB2_ID = "HuggingFaceFW/fineweb-2"
FINEWEB_ID = "HuggingFaceFW/fineweb"

TARGET_LANGUAGES = {
    "English": ["English (FineWeb)"],
    "Tamil": ["tam_Taml"],
    "Telugu": ["tel_Telu"],
    "Kannada": ["kan_Knda"],
    "Malayalam": ["mal_Mlym"],
    "Hindi": ["hin_Deva"],
    "Chinese": ["cmn_Hani", "cmn_Hans"],
    "Japanese": ["jpn_Jpan"],
}


def inspect_configs(dataset_id):

    print(f"\nChecking: {dataset_id}")

    try:
        configs = get_dataset_config_names(dataset_id)

        print("Configuration metadata retrieved successfully.")

        return set(configs)

    except Exception as error:

        print("Could not retrieve configurations.")
        print("Error:", error)

        return set()


def main():

    print("=" * 65)
    print("TCMB - CALIBRATION CORPUS AVAILABILITY CHECK")
    print("=" * 65)

    fineweb2_configs = inspect_configs(FINEWEB2_ID)

    fineweb_configs = inspect_configs(FINEWEB_ID)

    print("\n" + "=" * 65)
    print("LANGUAGE AVAILABILITY REPORT")
    print("=" * 65)

    for language, candidates in TARGET_LANGUAGES.items():

        print(f"\n{language}:")

        if language == "English":

            available = [
                config
                for config in ["default", "sample-10BT"]
                if config in fineweb_configs
            ]

            if available:
                print("FineWeb available:", available)
            else:
                print("FineWeb configuration needs verification.")

            continue

        available = [
            config
            for config in candidates
            if config in fineweb2_configs
        ]

        if available:
            print("FineWeb-2 available:", available)
        else:
            print("No exact configuration found.")

    print("\n" + "=" * 65)
    print("AVAILABLE CHINESE CONFIGURATIONS")
    print("=" * 65)

    chinese_configs = [
        config
        for config in fineweb2_configs
        if config.startswith("cmn_")
    ]

    print(chinese_configs)

    print("\nAvailability inspection completed.")
    print("No dataset contents were downloaded.")


if __name__ == "__main__":
    main()