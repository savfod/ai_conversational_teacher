"""Command-line interface for AI conversational teacher."""

import argparse
import json
import sys
from pathlib import Path

from config import (
    AppConfig,
    create_config_interactive,
)

from main import ConversationalTeacher


def _add_start_subparser(subparsers: argparse._SubParsersAction, name: str) -> None:
    """Add start subparser for the CLI.
    
    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the start command to.
        name (str): The name of the start command.
    """
    start_parser = subparsers.add_parser(name, help="Start a practice session")
    start_parser.add_argument("--config", type=str, help="Path to configuration file")


def _add_config_subparser(subparsers: argparse._SubParsersAction, name: str = "config") -> None:
    """Add configuration subparser for the CLI.
    
    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the config command to.
        name (str): The name of the config command. Defaults to "config".
    """
    config_parser = subparsers.add_parser(name, help="Manage configuration")
    config_parser.add_argument(
        "--create", action="store_true", help="Create a new configuration interactively"
    )
    config_parser.add_argument(
        "--output",
        type=str,
        default="config.json",
        help="Output path for configuration file",
    )
    config_parser.add_argument("--show", type=str, help="Show configuration from file")


def _add_stats_subparser(subparsers: argparse._SubParsersAction, name: str = "stats") -> None:
    """Add statistics subparser for the CLI.
    
    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the stats command to.
        name (str): The name of the stats command. Defaults to "stats".
    """
    stats_parser = subparsers.add_parser(name, help="View statistics")
    stats_parser.add_argument("--stats-file", type=str, help="Path to statistics file")


def _add_export_subparser(subparsers: argparse._SubParsersAction, name: str = "export") -> None:
    """Add export subparser for the CLI.
    
    Args:
        subparsers (argparse._SubParsersAction): The subparsers object to add the export command to.
        name (str): The name of the export command. Defaults to "export".
    """
    export_parser = subparsers.add_parser(name, help="Export vocabulary to Anki")
    export_parser.add_argument(
        "--stats-file", type=str, required=True, help="Path to statistics file"
    )
    export_parser.add_argument("--output", type=str, help="Output path for Anki export")



def main() -> None:
    """Main CLI entry point.
    
    Parses command line arguments and executes the appropriate command.
    Supports start, config, stats, and export commands.
    
    Commands:
        start: Start a practice session with optional config file
        config: Create or show configuration files
        stats: Display statistics from a statistics file
        export: Export vocabulary to Anki format
    """
    parser = argparse.ArgumentParser(
        description="AI Conversational Teacher - Practice languages with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default configuration
  python cli.py start
  
  # Create a custom configuration
  python cli.py config --create
  
  # Start with custom configuration
  python cli.py start --config my_config.json
  
  # Show statistics
  python cli.py stats --stats-file my_stats.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add subparsers using dedicated functions
    _add_start_subparser(subparsers, "start")
    _add_config_subparser(subparsers, "config")
    _add_stats_subparser(subparsers, "stats")
    _add_export_subparser(subparsers, "export")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle commands
    if args.command == "start":
        teacher = ConversationalTeacher(args.config)
        teacher.run_interactive()

    elif args.command == "config":
        if args.create:
            config = create_config_interactive()
            config.save_to_file(args.output)
            print(f"Configuration saved to: {args.output}")

        elif args.show:
            if not Path(args.show).exists():
                print(f"Error: Configuration file not found: {args.show}")
                sys.exit(1)

            config = AppConfig.load_from_file(args.show)
            print("\nCurrent Configuration:")
            print("=" * 60)
            print(config.model_dump_json(indent=2))
            print("=" * 60)

        else:
            print("Error: Please specify either --create or --show option.")
            print("Use 'python cli.py config --help' for more information.")

    elif args.command == "stats":
        if not args.stats_file or not Path(args.stats_file).exists():
            print("Error: Statistics file not found or not specified.")
            print("Use --stats-file to specify the statistics file.")
            sys.exit(1)

        from .statistics import StatisticsTracker
        
        stats = StatisticsTracker.load_from_file(args.stats_file)

        summary = stats.get_summary()
        print("\n" + "=" * 60)
        print("Statistics Summary")
        print("=" * 60)
        print(json.dumps(summary, indent=2, default=str))
        print("=" * 60)

    elif args.command == "export":
        if not Path(args.stats_file).exists():
            print(f"Error: Statistics file not found: {args.stats_file}")
            sys.exit(1)

        from .anki_exporter import AnkiExporter
        from .statistics import StatisticsTracker

        stats = StatisticsTracker.load_from_file(args.stats_file)

        if stats.vocabulary_stats.total_words == 0:
            print("No vocabulary to export.")
            sys.exit(0)

        # Determine language (use first word's context or default)
        language = "Unknown"
        if stats.vocabulary_stats.vocabulary_list:
            # We don't store language in stats, so use a default
            language = "Practice"

        exporter = AnkiExporter(language)
        output_path = args.output or exporter.create_export_filename()

        exporter.export_to_csv(stats.vocabulary_stats.vocabulary_list, output_path)


if __name__ == "__main__":
    main()
