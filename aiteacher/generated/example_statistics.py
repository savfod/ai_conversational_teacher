"""
Example: Tracking statistics and exporting to Anki
"""

from statistics import StatisticsTracker
from error_checker import GrammarError, VocabularyItem
from anki_exporter import AnkiExporter

# Create statistics tracker
stats = StatisticsTracker()

# Start a practice session
session_id = stats.start_session()
print(f"Session started: {session_id}")

# Record some activity
stats.record_message()
stats.record_message()

# Record errors
errors = [
    GrammarError(
        error_type="verb_conjugation",
        incorrect_text="yo es",
        corrected_text="yo soy",
        explanation="'Ser' conjugates as 'soy' for 'yo'",
        severity="major",
    ),
    GrammarError(
        error_type="article",
        incorrect_text="un casa",
        corrected_text="una casa",
        explanation="'Casa' is feminine, use 'una'",
        severity="medium",
    ),
]
stats.record_errors(errors)

# Record vocabulary
vocabulary = [
    VocabularyItem(
        word="casa", translation="house", context="Mi casa es grande", difficulty="easy"
    ),
    VocabularyItem(
        word="estudiante",
        translation="student",
        context="Soy estudiante",
        difficulty="easy",
    ),
    VocabularyItem(
        word="biblioteca",
        translation="library",
        context="Voy a la biblioteca",
        difficulty="medium",
    ),
]
stats.record_vocabulary(vocabulary)

# End session
stats.end_session()

# Get summary
summary = stats.get_summary()
print(f"\nSession Summary:")
print(f"Messages: {summary['total_messages']}")
print(f"Errors: {summary['error_statistics']['total_errors']}")
print(f"Words learned: {summary['vocabulary_statistics']['total_words']}")

# Save statistics
stats.save_to_file("my_stats.json")
print("\nStatistics saved to my_stats.json")

# Export vocabulary to Anki
exporter = AnkiExporter("Spanish")
exporter.export_to_csv(vocabulary, "spanish_vocabulary.csv")
print("Vocabulary exported to spanish_vocabulary.csv")

# Show what the CSV looks like
print("\nAnki CSV contents:")
with open("spanish_vocabulary.csv", "r", encoding="utf-8") as f:
    for line in f:
        print(f"  {line.strip()}")
