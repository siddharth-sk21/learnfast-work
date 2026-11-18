
import csv
import matplotlib.pyplot as plt
from pathlib import Path

results = []

with open("results/final_comparison.csv", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    results = list(reader)

models = [r["Model"] for r in results]

validation = [
    float(r["Validation Perplexity"]) for r in results
]

testing = [
    float(r["Test Perplexity"]) for r in results
]

x = range(len(models))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    [i - width / 2 for i in x],
    validation,
    width,
    label="Validation"
)

plt.bar(
    [i + width / 2 for i in x],
    testing,
    width,
    label="Test"
)

plt.yscale("log")
plt.xticks(list(x), models)
plt.ylabel("Perplexity (log scale)")
plt.title("Final Bigram vs Trigram Comparison")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/final_comparison.png",
    dpi=300
)

plt.show()

print("Final comparison graph saved!")
