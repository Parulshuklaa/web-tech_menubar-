#!/usr/bin/env python3

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "the",
    "to",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "you",
    "your",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [normalize_token(word) for word in words if word not in STOP_WORDS]


def normalize_token(token: str) -> str:
    for suffix in ("ing", "ed", "ly", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def vectorize(tokens: list[str]) -> Counter[str]:
    return Counter(tokens)


def cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0

    dot = sum(left[token] * right[token] for token in left if token in right)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True)
class Intent:
    name: str
    patterns: tuple[str, ...]
    response: str


INTENTS = (
    Intent(
        name="greeting",
        patterns=("hello", "hi there", "good morning", "hey"),
        response="Hello. How can I help you today?",
    ),
    Intent(
        name="about",
        patterns=("who are you", "what are you", "tell me about yourself"),
        response="I am a simple NLP chatbot built in Python using token matching and cosine similarity.",
    ),
    Intent(
        name="hours",
        patterns=("what are your hours", "when are you open", "opening hours"),
        response="I am always available because I run locally on your machine.",
    ),
    Intent(
        name="help",
        patterns=("can you help me", "i need support", "help"),
        response="I can answer simple questions, hold a short conversation, and demonstrate basic NLP intent matching.",
    ),
    Intent(
        name="portfolio",
        patterns=("tell me about this portfolio", "what projects are here", "show portfolio details"),
        response="This project currently contains a standalone NLP chatbot. You can extend it with your own portfolio data or project FAQs.",
    ),
    Intent(
        name="goodbye",
        patterns=("bye", "goodbye", "see you later", "exit"),
        response="Goodbye. Type another message anytime to continue.",
    ),
)


class NLPChatbot:
    def __init__(self, intents: tuple[Intent, ...]) -> None:
        self.intents = intents
        self.pattern_lookup = {
            pattern.lower(): intent.response
            for intent in intents
            for pattern in intent.patterns
        }
        self.intent_vectors = {
            intent.name: [vectorize(tokenize(pattern)) for pattern in intent.patterns]
            for intent in intents
        }

    def reply(self, message: str) -> str:
        clean_message = message.strip()
        if not clean_message:
            return "Please type a message so I can respond."

        exact_match = self.pattern_lookup.get(clean_message.lower())
        if exact_match:
            return exact_match

        if clean_message.lower() in {"quit", "exit"}:
            return "Session ended. Run the program again whenever you want to chat."

        user_vector = vectorize(tokenize(clean_message))
        best_intent = None
        best_score = 0.0

        for intent in self.intents:
            pattern_scores = [
                cosine_similarity(user_vector, pattern_vector)
                for pattern_vector in self.intent_vectors[intent.name]
            ]
            score = max(pattern_scores, default=0.0)
            if score > best_score:
                best_score = score
                best_intent = intent

        if best_intent and best_score >= 0.35:
            return best_intent.response

        return (
            "I did not fully understand that. Try asking about my purpose, help, hours, "
            "or say hello."
        )


def main() -> None:
    chatbot = NLPChatbot(INTENTS)
    print("NLP Chatbot")
    print("Type 'quit' to end the session.")

    while True:
        user_message = input("\nYou: ").strip()
        if user_message.lower() == "quit":
            print("Bot: Session ended. Goodbye.")
            break

        response = chatbot.reply(user_message)
        print(f"Bot: {response}")


if __name__ == "__main__":
    main()
