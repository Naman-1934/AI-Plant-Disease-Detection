"""
Performance metrics for the trained model
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

OUTPUT_DIR = Path("evaluation")

def save_confusion_matrix(true_labels, predicted_labels, class_names, ) -> None:
    """
    Generate and save the confusion matrix
    """

    # Create Matrix
    matrix = confusion_matrix(true_labels, predicted_labels,)

    # subplots = This creates an empty canvas.
    figure, axis = plt.subplots(figsize=(16, 16))

    # ConfusionMatrixDisplay = This converts the matrix into a beautiful figure.
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names,)

    # .plot = This draws cells, numbers, colors, labels onto the blank canvas.
    display.plot(ax=axis, xticks_rotation=90, colorbar=False,)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Instead of showing the figure, we save it.
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300,)

    plt.close()


def save_classification_report(true_labels, predicted_labels, class_names, ) -> None:
    """
    Generate and save the classification report.
    """

    # target_names = Without this you would see 0 1 2 3 4 Instead you'll get Apple___Black_rot Apple___Scab Corn___Healthy 
    # Much more readable.

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_file = OUTPUT_DIR / "classification_report.txt"

    with output_file.open("w", encoding="UTF-8",) as file:
        file.write(report)

    print(f"Successfully report saved to {output_file}")
