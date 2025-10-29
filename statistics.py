"""Statistics tracking for language learning progress."""

from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from error_checker import GrammarError, VocabularyItem


class ErrorStatistics(BaseModel):
    """Statistics about errors made during practice."""
    
    total_errors: int = Field(default=0)
    errors_by_type: Dict[str, int] = Field(default_factory=dict)
    errors_by_severity: Dict[str, int] = Field(default_factory=dict)
    recent_errors: List[GrammarError] = Field(default_factory=list, max_length=100)


class VocabularyStatistics(BaseModel):
    """Statistics about vocabulary encountered."""
    
    total_words: int = Field(default=0)
    words_by_difficulty: Dict[str, int] = Field(default_factory=dict)
    vocabulary_list: List[VocabularyItem] = Field(default_factory=list)


class SessionStatistics(BaseModel):
    """Statistics for a practice session."""
    
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    messages_exchanged: int = Field(default=0)
    errors_made: int = Field(default=0)
    words_learned: int = Field(default=0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StatisticsTracker:
    """Tracks statistics for language learning progress."""
    
    def __init__(self):
        """Initialize statistics tracker."""
        self.error_stats = ErrorStatistics()
        self.vocabulary_stats = VocabularyStatistics()
        self.sessions: List[SessionStatistics] = []
        self.current_session: Optional[SessionStatistics] = None
    
    def start_session(self) -> str:
        """Start a new practice session.
        
        Returns:
            Session ID
        """
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = SessionStatistics(
            session_id=session_id,
            start_time=datetime.now()
        )
        self.sessions.append(self.current_session)
        return session_id
    
    def end_session(self) -> None:
        """End the current practice session."""
        if self.current_session:
            self.current_session.end_time = datetime.now()
            self.current_session = None
    
    def record_message(self) -> None:
        """Record a message exchange."""
        if self.current_session:
            self.current_session.messages_exchanged += 1
    
    def record_errors(self, errors: List[GrammarError]) -> None:
        """Record errors from error checking.
        
        Args:
            errors: List of grammar errors detected
        """
        if not errors:
            return
        
        self.error_stats.total_errors += len(errors)
        
        if self.current_session:
            self.current_session.errors_made += len(errors)
        
        for error in errors:
            # Track by type
            error_type = error.error_type
            self.error_stats.errors_by_type[error_type] = \
                self.error_stats.errors_by_type.get(error_type, 0) + 1
            
            # Track by severity
            severity = error.severity
            self.error_stats.errors_by_severity[severity] = \
                self.error_stats.errors_by_severity.get(severity, 0) + 1
            
            # Keep recent errors
            self.error_stats.recent_errors.append(error)
            if len(self.error_stats.recent_errors) > 100:
                self.error_stats.recent_errors.pop(0)
    
    def record_vocabulary(self, vocabulary: List[VocabularyItem]) -> None:
        """Record new vocabulary encountered.
        
        Args:
            vocabulary: List of vocabulary items
        """
        if not vocabulary:
            return
        
        if self.current_session:
            self.current_session.words_learned += len(vocabulary)
        
        for item in vocabulary:
            # Check if word already exists
            if not any(v.word == item.word for v in self.vocabulary_stats.vocabulary_list):
                self.vocabulary_stats.total_words += 1
                self.vocabulary_stats.vocabulary_list.append(item)
                
                # Track by difficulty
                difficulty = item.difficulty
                self.vocabulary_stats.words_by_difficulty[difficulty] = \
                    self.vocabulary_stats.words_by_difficulty.get(difficulty, 0) + 1
    
    def get_summary(self) -> Dict:
        """Get a summary of statistics.
        
        Returns:
            Dictionary with statistics summary
        """
        return {
            "total_sessions": len(self.sessions),
            "total_messages": sum(s.messages_exchanged for s in self.sessions),
            "error_statistics": {
                "total_errors": self.error_stats.total_errors,
                "by_type": self.error_stats.errors_by_type,
                "by_severity": self.error_stats.errors_by_severity,
            },
            "vocabulary_statistics": {
                "total_words": self.vocabulary_stats.total_words,
                "by_difficulty": self.vocabulary_stats.words_by_difficulty,
            },
            "current_session": {
                "id": self.current_session.session_id if self.current_session else None,
                "messages": self.current_session.messages_exchanged if self.current_session else 0,
                "errors": self.current_session.errors_made if self.current_session else 0,
                "words": self.current_session.words_learned if self.current_session else 0,
            } if self.current_session else None
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save statistics to a file.
        
        Args:
            filepath: Path to save statistics to
        """
        import json
        
        data = {
            "error_stats": self.error_stats.model_dump(),
            "vocabulary_stats": self.vocabulary_stats.model_dump(),
            "sessions": [s.model_dump() for s in self.sessions],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "StatisticsTracker":
        """Load statistics from a file.
        
        Args:
            filepath: Path to load statistics from
            
        Returns:
            StatisticsTracker instance
        """
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        tracker = cls()
        tracker.error_stats = ErrorStatistics(**data["error_stats"])
        tracker.vocabulary_stats = VocabularyStatistics(**data["vocabulary_stats"])
        tracker.sessions = [SessionStatistics(**s) for s in data["sessions"]]
        
        return tracker
