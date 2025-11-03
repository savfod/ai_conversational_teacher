"""Main application for AI conversational teacher."""

import os
from statistics import StatisticsTracker
from typing import Optional

from anki_exporter import AnkiExporter
from config import AppConfig
from error_checker import ErrorChecker, ErrorCheckResult
from voice_interface import VoiceInterface


class ConversationalTeacher:
    """Main application for AI conversational language teacher."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the conversational teacher.

        Args:
            config_path: Path to configuration file. If None, uses defaults.
        """
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = AppConfig.load_from_file(config_path)
        else:
            self.config = AppConfig()

        # Initialize components
        self.voice_interface = VoiceInterface(
            enabled=self.config.voice.enabled, auto_listen=self.config.voice.auto_listen
        )

        self.error_checker = ErrorChecker(
            language=self.config.language.language,
            level=self.config.language.level,
            native_language=self.config.language.native_language,
        )

        self.statistics = StatisticsTracker()

        self.anki_exporter = AnkiExporter(language=self.config.language.language)

        self.conversation_history = []

    def start_session(self) -> None:
        """Start a new practice session."""
        session_id = self.statistics.start_session()
        print(f"\n{'=' * 60}")
        print(f"Welcome to AI Conversational Teacher!")
        print(f"{'=' * 60}")
        print(f"Session ID: {session_id}")
        print(f"Language: {self.config.language.language}")
        print(f"Level: {self.config.language.level}")
        print(
            f"Voice Interface: {'Enabled' if self.config.voice.enabled else 'Disabled'}"
        )
        print(f"{'=' * 60}\n")

    def end_session(self) -> None:
        """End the current practice session."""
        self.statistics.end_session()

        # Show statistics
        summary = self.statistics.get_summary()
        print(f"\n{'=' * 60}")
        print("Session Summary")
        print(f"{'=' * 60}")
        print(f"Messages exchanged: {summary['total_messages']}")
        print(f"Total errors: {summary['error_statistics']['total_errors']}")
        print(f"Words learned: {summary['vocabulary_statistics']['total_words']}")
        print(f"{'=' * 60}\n")

        # Export to Anki if enabled and vocabulary exists
        if (
            self.config.statistics.export_to_anki
            and self.statistics.vocabulary_stats.total_words > 0
        ):
            self.export_vocabulary()

    def get_user_input(self) -> Optional[str]:
        """Get input from user (voice or text).

        Returns:
            User input text or None if quit
        """
        # Try voice input first if enabled
        if self.config.voice.enabled:
            voice_input = self.voice_interface.get_voice_input()
            if voice_input:
                return voice_input

        # Fall back to text input
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit", "q"]:
                return None
            return user_input
        except (EOFError, KeyboardInterrupt):
            return None

    def process_input(self, user_input: str) -> ErrorCheckResult:
        """Process user input and check for errors.

        Args:
            user_input: User's text input

        Returns:
            Error check result
        """
        self.statistics.record_message()

        # Check for errors
        result = self.error_checker.check_errors(user_input)

        # Record statistics
        if self.config.statistics.track_errors and result.has_errors:
            self.statistics.record_errors(result.errors)

        if self.config.statistics.track_vocabulary and result.vocabulary:
            self.statistics.record_vocabulary(result.vocabulary)

        return result

    def generate_response(self, user_input: str) -> str:
        """Generate AI response to user input.

        Args:
            user_input: User's input text

        Returns:
            AI response text

        Stub: In production, this would call OpenAI API.
        """
        # This is a stub implementation
        # In production, this would:
        # 1. Use OpenAI API to generate contextual response
        # 2. Consider conversation history
        # 3. Adjust complexity based on user's level

        return (
            f"[AI Response - STUB MODE]\n"
            f"Thank you for practicing {self.config.language.language} with me! "
            f"In a full implementation, I would respond contextually to your message.\n"
            f"Your message: '{user_input}'"
        )

    def run_interactive(self) -> None:
        """Run interactive conversation loop."""
        self.start_session()

        print("Start practicing! (Type 'quit', 'exit', or 'q' to end session)\n")

        try:
            while True:
                # Get user input
                user_input = self.get_user_input()
                if user_input is None:
                    break

                if not user_input:
                    continue

                # Process input and check for errors
                error_result = self.process_input(user_input)

                # Show errors if any
                if error_result.has_errors:
                    print("\n[Error Feedback]")
                    for i, error in enumerate(error_result.errors, 1):
                        print(
                            f"  {i}. {error.error_type}: '{error.incorrect_text}' → '{error.corrected_text}'"
                        )
                        print(f"     {error.explanation}")
                    print()

                # Generate and show AI response
                response = self.generate_response(user_input)
                print(f"\nAI: {response}\n")

                # Speak response if voice enabled
                if self.config.voice.enabled:
                    self.voice_interface.speak(response)

                # Auto-listen if enabled
                if self.config.voice.auto_listen:
                    self.voice_interface.start_listening()

        except KeyboardInterrupt:
            print("\n\nSession interrupted.")

        finally:
            self.end_session()

    def export_vocabulary(self, filepath: Optional[str] = None) -> None:
        """Export vocabulary to Anki format.

        Args:
            filepath: Optional custom filepath. If None, generates automatic filename.
        """
        if self.statistics.vocabulary_stats.total_words == 0:
            print("No vocabulary to export.")
            return

        if filepath is None:
            filepath = self.anki_exporter.create_export_filename()

        self.anki_exporter.export_to_csv(
            self.statistics.vocabulary_stats.vocabulary_list, filepath
        )

    def show_statistics(self) -> None:
        """Display current statistics."""
        summary = self.statistics.get_summary()

        print("\n" + "=" * 60)
        print("Statistics")
        print("=" * 60)

        print("\nError Statistics:")
        print(f"  Total errors: {summary['error_statistics']['total_errors']}")
        if summary["error_statistics"]["by_type"]:
            print("  By type:")
            for error_type, count in summary["error_statistics"]["by_type"].items():
                print(f"    - {error_type}: {count}")
        if summary["error_statistics"]["by_severity"]:
            print("  By severity:")
            for severity, count in summary["error_statistics"]["by_severity"].items():
                print(f"    - {severity}: {count}")

        print("\nVocabulary Statistics:")
        print(f"  Total words: {summary['vocabulary_statistics']['total_words']}")
        if summary["vocabulary_statistics"]["by_difficulty"]:
            print("  By difficulty:")
            for difficulty, count in summary["vocabulary_statistics"][
                "by_difficulty"
            ].items():
                print(f"    - {difficulty}: {count}")

        print(f"\nTotal sessions: {summary['total_sessions']}")
        print(f"Total messages: {summary['total_messages']}")
        print("=" * 60 + "\n")


def main():
    """Main entry point."""
    import sys

    # Parse command line arguments
    config_path = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    # Create and run teacher
    teacher = ConversationalTeacher(config_path)

    # Run interactive session
    teacher.run_interactive()


if __name__ == "__main__":
    main()
