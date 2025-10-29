"""Error checking with structured LLM output."""

from typing import List, Optional
from pydantic import BaseModel, Field


class GrammarError(BaseModel):
    """Represents a grammar error detected in user input."""
    
    error_type: str = Field(
        description="Type of error (e.g., 'verb_conjugation', 'article', 'gender', 'word_order')"
    )
    
    incorrect_text: str = Field(
        description="The incorrect text from user input"
    )
    
    corrected_text: str = Field(
        description="The corrected version"
    )
    
    explanation: str = Field(
        description="Explanation of the error in user's native language"
    )
    
    severity: str = Field(
        default="medium",
        description="Error severity: 'minor', 'medium', 'major'"
    )


class VocabularyItem(BaseModel):
    """Represents a vocabulary word encountered."""
    
    word: str = Field(
        description="The word in target language"
    )
    
    translation: str = Field(
        description="Translation to native language"
    )
    
    context: str = Field(
        description="Context/example sentence where word was used"
    )
    
    difficulty: str = Field(
        default="medium",
        description="Word difficulty: 'easy', 'medium', 'hard'"
    )


class ErrorCheckResult(BaseModel):
    """Structured result from error checking."""
    
    has_errors: bool = Field(
        description="Whether any errors were detected"
    )
    
    errors: List[GrammarError] = Field(
        default_factory=list,
        description="List of detected errors"
    )
    
    vocabulary: List[VocabularyItem] = Field(
        default_factory=list,
        description="New vocabulary items used"
    )
    
    overall_feedback: str = Field(
        description="General feedback on the response"
    )
    
    fluency_score: Optional[float] = Field(
        default=None,
        description="Fluency score from 0.0 to 1.0"
    )


class ErrorChecker:
    """Checks user input for errors using structured LLM output.
    
    This is a stub implementation. In production, this would use OpenAI's
    structured output feature with JSON mode or function calling.
    """
    
    def __init__(self, language: str, level: str, native_language: str):
        """Initialize error checker.
        
        Args:
            language: Target language being practiced
            level: User's proficiency level
            native_language: User's native language for explanations
        """
        self.language = language
        self.level = level
        self.native_language = native_language
    
    def check_errors(self, user_input: str, context: str = "") -> ErrorCheckResult:
        """Check user input for errors.
        
        Args:
            user_input: User's text to check
            context: Optional context of the conversation
            
        Returns:
            Structured error check result
            
        Stub: In production, this would call OpenAI API with structured output.
        """
        # Stub implementation - return a sample result
        print(f"[Error Checker] Analyzing: '{user_input}'")
        
        # In a real implementation, this would:
        # 1. Call OpenAI API with response_format={"type": "json_object"}
        # 2. Use a prompt that asks for JSON output matching ErrorCheckResult schema
        # 3. Parse and validate the JSON response using Pydantic
        
        # Return empty result for stub
        return ErrorCheckResult(
            has_errors=False,
            errors=[],
            vocabulary=[],
            overall_feedback="Analysis complete (stub mode - no actual checking performed)",
            fluency_score=None
        )
    
    def get_correction_prompt(self, user_input: str) -> str:
        """Generate a prompt for error correction.
        
        Args:
            user_input: User's input text
            
        Returns:
            Formatted prompt for LLM
        """
        return f"""You are a {self.language} teacher helping a {self.level} student.
Analyze the following text and return a JSON object with error corrections and vocabulary.

Student's text: "{user_input}"

Return JSON matching this structure:
{{
    "has_errors": boolean,
    "errors": [
        {{
            "error_type": "string",
            "incorrect_text": "string",
            "corrected_text": "string",
            "explanation": "string in {self.native_language}",
            "severity": "minor|medium|major"
        }}
    ],
    "vocabulary": [
        {{
            "word": "string",
            "translation": "string in {self.native_language}",
            "context": "string",
            "difficulty": "easy|medium|hard"
        }}
    ],
    "overall_feedback": "string in {self.native_language}",
    "fluency_score": 0.0 to 1.0
}}"""
