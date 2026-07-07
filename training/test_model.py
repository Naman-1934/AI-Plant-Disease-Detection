"""
Test model creation
"""

from training.data_loader import DataLoader
from training.model import PlantDiseaseModel

def main() -> None:
    """
    Test the model.
    """

    train_ds, _, class_names = DataLoader.load_dataset()

    model = PlantDiseaseModel.build(
        num_classes = len(class_names)
    )

    model.summary()

if __name__ == "__main__":
    main()