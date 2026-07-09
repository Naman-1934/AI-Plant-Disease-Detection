"""
Model training script.
"""
import tensorflow as tf
from training.call_backs import CallbackFactory
from training.config import EPOCHS
from training.data_loader import DataLoader
from training.model import PlantDiseaseModel

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main() -> None:
    """
    Train the plant disease classification model.
    """

    print("Loading Dataset...")

    train_dataset, validation_dataset, class_names = DataLoader.load_dataset()

    print(f"Number of Classes: {len(class_names)}")

    print("Building Model...")

    model = PlantDiseaseModel.build(num_classes=len(class_names))

    print("Starting training...\n")

    print("Checking dataset...")

    for images, labels in train_dataset.take(1):
        print(images.shape)
        print(labels.shape)
    print("Dataset is Ok")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=CallbackFactory.get_callbacks(),
    )

    print("\nTraining completed")

    print(history.history.keys())

if __name__ == "__main__":
    main()