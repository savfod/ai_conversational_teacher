"""Anki export functionality for vocabulary learning."""

import csv
from datetime import datetime
from typing import List

from error_checker import VocabularyItem


class AnkiExporter:
    """Export vocabulary to Anki-compatible format.

    Anki can import CSV files with tab-separated values.
    Basic format: Front, Back, Tags
    """

    def __init__(self, language: str):
        """Initialize Anki exporter.

        Args:
            language: Target language being practiced
        """
        self.language = language

    def export_to_csv(
        self,
        vocabulary: List[VocabularyItem],
        filepath: str,
        include_context: bool = True,
    ) -> None:
        """Export vocabulary to Anki-compatible CSV file.

        Args:
            vocabulary: List of vocabulary items to export
            filepath: Path to save CSV file
            include_context: Whether to include context in the back of card
        """
        if not vocabulary:
            print("No vocabulary to export.")
            return

        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            # Use tab as delimiter (Anki default)
            writer = csv.writer(csvfile, delimiter="\t")

            # Write header (optional, can be removed if Anki deck doesn't need it)
            # writer.writerow(['Front', 'Back', 'Tags'])

            for item in vocabulary:
                front = item.word

                # Build back of card
                back_parts = [item.translation]
                if include_context and item.context:
                    back_parts.append(f"<br><br><i>Context: {item.context}</i>")
                back = "".join(back_parts)

                # Create tags
                tags = f"{self.language} {item.difficulty}".lower()

                writer.writerow([front, back, tags])

        print(f"Exported {len(vocabulary)} words to {filepath}")

    def export_to_apkg(
        self,
        vocabulary: List[VocabularyItem],
        filepath: str,
        deck_name: str = "Language Practice",
    ) -> None:
        """Export vocabulary to Anki package (.apkg) format.

        Args:
            vocabulary: List of vocabulary items to export
            filepath: Path to save .apkg file
            deck_name: Name of the Anki deck

        Note: This is a stub. Real implementation would require the genanki library.
        """
        print("[Anki Exporter] STUB: .apkg export not implemented yet.")
        print(f"Would export {len(vocabulary)} words to {filepath}")
        print(f"Deck name: {deck_name}")
        print("Use export_to_csv() for now, which creates a file importable by Anki.")

    def create_export_filename(self, prefix: str = "vocabulary") -> str:
        """Generate a filename for export.

        Args:
            prefix: Prefix for filename

        Returns:
            Filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{self.language.lower()}_{timestamp}.csv"
