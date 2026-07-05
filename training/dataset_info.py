"""
Display dataset information.
"""

from training.utils.dataset_validator import DatasetValidator

def main() -> None:
    """
    Entry Point.
    """

    DatasetValidator.validates()

if __name__ == "__main__":
    main()