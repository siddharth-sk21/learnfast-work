
import re
import math
import random
from collections import Counter


class NgramLM:

    def __init__(self, n=2, k=1.0, min_count=2):
        self.n = n
        self.k = k
        self.min_count = min_count

        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()

    def tokenize(self, text):
        return re.findall(r"\b\w+\b|[^\w\s]", text.lower())

    def fit(self, text):
        sentences = [
            self.tokenize(line)
            for line in text.splitlines()
            if line.strip()
        ]

        word_counts = Counter(
            word for sentence in sentences for word in sentence
        )

        self.vocabulary = {
            word for word, count in word_counts.items()
            if count >= self.min_count
        }

        self.vocabulary.update({"<UNK>", "</s>"})

        self.ngram_counts.clear()
        self.context_counts.clear()

        for sentence in sentences:
            tokens = [
                word if word in self.vocabulary else "<UNK>"
                for word in sentence
            ]

            if self.n > 1:
                tokens = ["<s>"] * (self.n - 1) + tokens

            tokens.append("</s>")

            for i in range(self.n - 1, len(tokens)):
                ngram = tuple(tokens[i - self.n + 1:i + 1])
                context = ngram[:-1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        print(f"Model trained with {self.n}-grams")
        print(f"Vocabulary size: {len(self.vocabulary)}")
        print(f"Unique n-grams: {len(self.ngram_counts)}")

    def suggest_next(self, context, top_k=5):
        tokens = self.tokenize(context)

        tokens = [
            word if word in self.vocabulary else "<UNK>"
            for word in tokens
        ]

        if self.n == 1:
            context_tuple = ()
        else:
            tokens = ["<s>"] * max(0, self.n - 1 - len(tokens)) + tokens
            context_tuple = tuple(tokens[-(self.n - 1):])

        vocab_size = len(self.vocabulary)
        candidates = []

        for word in self.vocabulary:
            if word == "<s>":
                continue

            ngram = context_tuple + (word,)

            probability = (
                (self.ngram_counts[ngram] + self.k) /
                (self.context_counts[context_tuple] + self.k * vocab_size)
            )

            candidates.append((word, probability))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    
    def generate(self, prompt, max_words=30, top_k=5, min_words=5):
        tokens = self.tokenize(prompt)
        generated_count = 0

        for _ in range(max_words):
            context = " ".join(tokens)

            candidates = self.suggest_next(
                context, top_k=len(self.vocabulary)
            )

            candidates = [
                (word, prob)
                for word, prob in candidates
                if word != "<UNK>"
            ]

            # Avoid ending the generation too early
            if generated_count < min_words:
                candidates = [
                    (word, prob)
                    for word, prob in candidates
                    if word != "</s>"
                ]

            candidates = candidates[:top_k]

            if not candidates:
                break

            words, probabilities = zip(*candidates)

            next_word = random.choices(
                words, weights=probabilities, k=1
            )[0]

            if next_word == "</s>":
                break

            tokens.append(next_word)
            generated_count += 1

            # Stop when a sentence-ending punctuation mark appears
            if next_word in {".", "!", "?"}:
                break

        generated = " ".join(tokens)
        return re.sub(r"\s+([,.;:!?])", r"\1", generated)


    def perplexity(self, text):
        sentences = [
            self.tokenize(line)
            for line in text.splitlines()
            if line.strip()
        ]

        log_probability = 0.0
        total = 0
        vocab_size = len(self.vocabulary)

        for sentence in sentences:
            tokens = [
                word if word in self.vocabulary else "<UNK>"
                for word in sentence
            ]

            if self.n > 1:
                tokens = ["<s>"] * (self.n - 1) + tokens

            tokens.append("</s>")

            for i in range(self.n - 1, len(tokens)):
                ngram = tuple(tokens[i - self.n + 1:i + 1])
                context = ngram[:-1]
                word = ngram[-1]

                probability = (
                    (self.ngram_counts[ngram] + self.k) /
                    (self.context_counts[context] + self.k * vocab_size)
                )

                log_probability += math.log(probability)
                total += 1

        if total == 0:
            return float("inf")

        return math.exp(-log_probability / total)
