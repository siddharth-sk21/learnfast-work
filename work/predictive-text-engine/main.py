
from pathlib import Path
import csv
from ngram_lm import NgramLM

# Load datasets
train_text = Path("data/train.txt").read_text(encoding="utf-8")
val_text = Path("data/validation.txt").read_text(encoding="utf-8")
test_text = Path("data/test.txt").read_text(encoding="utf-8")

smoothing_values = [0.1, 0.5, 1.0]
final_results = []

# Tune each model using validation data
for n in [2, 3]:

    print(f"\n========== {n}-GRAM EXPERIMENT ==========")

    best_k = None
    best_val_ppl = float("inf")

    for k in smoothing_values:

        model = NgramLM(n=n, k=k, min_count=2)
        model.fit(train_text)

        val_ppl = model.perplexity(val_text)

        print(f"k={k} | Validation perplexity={val_ppl:.2f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_k = k

    # Evaluate selected configuration on test data
    print(f"\nBest smoothing value: {best_k}")

    best_model = NgramLM(n=n, k=best_k, min_count=2)
    best_model.fit(train_text)

    test_ppl = best_model.perplexity(test_text)

    print(f"Final test perplexity: {test_ppl:.2f}")

    final_results.append({
        "Model": f"{n}-gram",
        "Best Smoothing": best_k,
        "Validation Perplexity": round(best_val_ppl, 4),
        "Test Perplexity": round(test_ppl, 4)
    })

# Display final comparison
print("\n========== FINAL MODEL COMPARISON ==========")

for result in final_results:
    print(result)

# Save results
Path("results").mkdir(exist_ok=True)

with open(
    "results/final_comparison.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file, fieldnames=final_results[0].keys()
    )
    writer.writeheader()
    writer.writerows(final_results)

print("\nFinal results saved to results/final_comparison.csv")
