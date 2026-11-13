import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core import train_v3, BANKING77_PATH


OUTPUT = Path("evaluation_results")
OUTPUT.mkdir(exist_ok=True)

print("\n=== SUPPORTMIND AI: ML EVALUATION ===")

# 1. Train the exact model used by Streamlit.
print("\nTraining BANKING77 classifier...")
model, stats = train_v3(use_banking77=True)

# 2. Load official held-out test examples.
data = json.loads(BANKING77_PATH.read_text())
test = data["test"]

X_test = [row["text"] for row in test]
y_true = [row["category"] for row in test]

print("Evaluating on", len(X_test), "test samples...")

# 3. Predict using the trained pipeline.
y_pred = model.predict(X_test)

# 4. Calculate metrics.
summary = {
    "dataset": stats["dataset"],
    "train_samples": stats["train_count"],
    "test_samples": len(y_true),
    "classes": stats["classes"],
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "macro_precision": float(
        precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
    ),
    "macro_recall": float(
        recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
    ),
    "macro_f1": float(
        f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0
        )
    ),
    "weighted_f1": float(
        f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0
        )
    ),
}

print("\n=== OVERALL RESULTS ===")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# 5. Save summary.
(OUTPUT / "evaluation_summary.json").write_text(
    json.dumps(summary, indent=2)
)

# 6. Per-class report.
report = classification_report(
    y_true,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).T
report_df.to_csv(OUTPUT / "classification_report.csv")

# 7. Save individual predictions.
predictions_df = pd.DataFrame({
    "text": X_test,
    "true_intent": y_true,
    "predicted_intent": y_pred,
    "correct": [
        actual == predicted
        for actual, predicted in zip(y_true, y_pred)
    ]
})

predictions_df.to_csv(
    OUTPUT / "test_predictions.csv",
    index=False
)

# 8. Analyze mistakes.
errors = predictions_df[
    ~predictions_df["correct"]
].copy()

errors.to_csv(
    OUTPUT / "misclassified_examples.csv",
    index=False
)

print("\nTotal incorrect predictions:", len(errors))

# 9. Find most common confusions.
confusions = (
    errors.groupby(
        ["true_intent", "predicted_intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

confusions.to_csv(
    OUTPUT / "top_confusions.csv",
    index=False
)

print("\n=== TOP 10 CONFUSED INTENT PAIRS ===")
print(confusions.head(10).to_string(index=False))

# 10. Plot confusion matrix.
labels = sorted(set(y_true))
cm = confusion_matrix(
    y_true, y_pred, labels=labels
)

fig, ax = plt.subplots(figsize=(18, 16))
im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

ax.set_title("BANKING77 Confusion Matrix")
ax.set_xlabel("Predicted intent")
ax.set_ylabel("True intent")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))

ax.set_xticklabels(labels, rotation=90, fontsize=5)
ax.set_yticklabels(labels, fontsize=5)

fig.colorbar(im, ax=ax)
fig.tight_layout()

fig.savefig(
    OUTPUT / "confusion_matrix.png",
    dpi=180
)
plt.close(fig)

# 11. Plot weakest classes by F1.
classes_df = report_df.loc[
    labels, ["precision", "recall", "f1-score", "support"]
]

weakest = classes_df.sort_values(
    "f1-score"
).head(15)

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(
    weakest.index,
    weakest["f1-score"]
)
ax.set_xlim(0, 1)
ax.set_xlabel("F1 score")
ax.set_title("15 Lowest-F1 BANKING77 Intents")
ax.invert_yaxis()
fig.tight_layout()

fig.savefig(
    OUTPUT / "weakest_intents.png",
    dpi=180
)
plt.close(fig)

print("\n=== EVALUATION COMPLETE ===")
print("Results saved in:", OUTPUT.resolve())
