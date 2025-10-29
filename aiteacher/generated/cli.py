"""Command-line interface for AI conversational teacher."""

import argparse
import sys
import json
from pathlib import Path
from config import AppConfig, LanguageConfig, VoiceConfig, StatisticsConfig
from main import ConversationalTeacher


def create_config_interactive() -> AppConfig:
    """Create configuration interactively."""
    print("\nConfiguration Setup")
    print("=" * 60)
    
    # Language settings
    language = input("Target language (e.g., Spanish, French, German) [Spanish]: ").strip() or "Spanish"
    level = input("Your level (beginner/intermediate/advanced) [beginner]: ").strip() or "beginner"
    native_language = input("Your native language [English]: ").strip() or "English"
    
    # Voice settings
    voice_enabled_input = input("Enable voice interface? (y/n) [n]: ").strip().lower()
    voice_enabled = voice_enabled_input in ['y', 'yes']
    
    auto_listen = False
    if voice_enabled:
        auto_listen_input = input("Auto-listen after AI responds? (y/n) [n]: ").strip().lower()
        auto_listen = auto_listen_input in ['y', 'yes']
    
    # Statistics settings
    track_errors_input = input("Track errors? (y/n) [y]: ").strip().lower()
    track_errors = track_errors_input not in ['n', 'no']
    
    track_vocabulary_input = input("Track vocabulary? (y/n) [y]: ").strip().lower()
    track_vocabulary = track_vocabulary_input not in ['n', 'no']
    
    export_to_anki_input = input("Enable Anki export? (y/n) [y]: ").strip().lower()
    export_to_anki = export_to_anki_input not in ['n', 'no']
    
    # OpenAI API key (optional for now)
    api_key = input("OpenAI API key (optional, press Enter to skip): ").strip()
    
    # Create config
    config = AppConfig(
        language=LanguageConfig(
            language=language,
            level=level,
            native_language=native_language
        ),
        voice=VoiceConfig(
            enabled=voice_enabled,
            auto_listen=auto_listen
        ),
        statistics=StatisticsConfig(
            track_errors=track_errors,
            track_vocabulary=track_vocabulary,
            export_to_anki=export_to_anki
        ),
        openai_api_key=api_key
    )
    
    print("\nConfiguration created successfully!")
    return config


def main():
    """Main CLI entry point."""
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
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start a practice session')
    start_parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument(
        '--create',
        action='store_true',
        help='Create a new configuration interactively'
    )
    config_parser.add_argument(
        '--output',
        type=str,
        default='config.json',
        help='Output path for configuration file'
    )
    config_parser.add_argument(
        '--show',
        type=str,
        help='Show configuration from file'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='View statistics')
    stats_parser.add_argument(
        '--stats-file',
        type=str,
        help='Path to statistics file'
    )
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export vocabulary to Anki')
    export_parser.add_argument(
        '--stats-file',
        type=str,
        required=True,
        help='Path to statistics file'
    )
    export_parser.add_argument(
        '--output',
        type=str,
        help='Output path for Anki export'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle commands
    if args.command == 'start':
        teacher = ConversationalTeacher(args.config)
        teacher.run_interactive()
    
    elif args.command == 'config':
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
            config_parser.print_help()
    
    elif args.command == 'stats':
        if not args.stats_file or not Path(args.stats_file).exists():
            print("Error: Statistics file not found or not specified.")
            print("Use --stats-file to specify the statistics file.")
            sys.exit(1)
        
        from statistics import StatisticsTracker
        stats = StatisticsTracker.load_from_file(args.stats_file)
        
        summary = stats.get_summary()
        print("\n" + "=" * 60)
        print("Statistics Summary")
        print("=" * 60)
        print(json.dumps(summary, indent=2, default=str))
        print("=" * 60)
    
    elif args.command == 'export':
        if not Path(args.stats_file).exists():
            print(f"Error: Statistics file not found: {args.stats_file}")
            sys.exit(1)
        
        from statistics import StatisticsTracker
        from anki_exporter import AnkiExporter
        
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
        
        exporter.export_to_csv(
            stats.vocabulary_stats.vocabulary_list,
            output_path
        )


if __name__ == "__main__":
    main()
