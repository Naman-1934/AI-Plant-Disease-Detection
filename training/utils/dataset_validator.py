"""
Dataset Validation utilitites.
"""

from pathlib import Path

from training.config import DATASET_DIR

class DatasetValidator:
    """
    Validates the   PlantVillage dataset.
    """

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
    }

    @classmethod
    def validates(cls) -> None:
        """
        Validate dataset structure.
        """

        if not DATASET_DIR.exists():
            raise FileNotFoundError(f"Dataset directory not found:\n{DATASET_DIR}")
        
        
        # is_dir = Whether this path is a directory.
        class_folders = sorted(
            folder
            for folder in DATASET_DIR.iterdir()
            if folder.is_dir()
        )

        if not class_folders:
            raise ValueError("No class folders found.")
        
        print(f"\nFound {len(class_folders)} classes.\n")

        total_images = 0

        for folder in class_folders:

            images = [
                file
                for file in folder.iterdir()
                if file.suffix.lower() in cls.SUPPORTED_EXTENSIONS
            ]

            print(f"{folder.name:<45} len(images):>5")

            total_images += len(images)

        print("-" * 65)

        print(f"Total Images: {total_images}")

        print("\nDataset validation successful.")