"""
Example: Using the error checker with structured output
"""

from error_checker import ErrorChecker, GrammarError, VocabularyItem

# Create error checker for Spanish
checker = ErrorChecker(
    language="Spanish",
    level="intermediate",
    native_language="English"
)

# Check a sentence (this is a stub - in production it would call OpenAI API)
result = checker.check_errors("Yo es un estudiante")

print(f"Has errors: {result.has_errors}")
print(f"Number of errors: {len(result.errors)}")

# In production, this would return actual errors like:
# GrammarError(
#     error_type="verb_conjugation",
#     incorrect_text="Yo es",
#     corrected_text="Yo soy",
#     explanation="The verb 'ser' conjugates as 'soy' for 'yo'",
#     severity="major"
# )

# Example of what the LLM prompt looks like
prompt = checker.get_correction_prompt("Yo es un estudiante")
print("\nExample prompt for LLM:")
print(prompt)
