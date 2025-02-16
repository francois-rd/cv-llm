def scrub(text: str) -> str:
    """Removes '{' and '}' characters from the text, which cause langchain formatting errors."""
    return text.replace("{", "").replace("}", "")
