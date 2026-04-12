"""
Shared Unicode sanitizer — strips hidden characters that can carry
steganographic prompt injection payloads.

Used by: waechter.py, vault_sanitizer.py
"""

import re

HIDDEN_CHAR_PATTERN = re.compile(
    "[\u200b-\u200f\ufeff\u00ad"      # Zero-Width Characters
    "\u200c\u200d"                      # Zero-Width Joiners
    "\ufe00-\ufe0f"                     # Variation Selectors
    "\U000e0000-\U000e007f"             # Tag Characters
    "\u202a-\u202e\u2066-\u2069"        # Directional Overrides
    "\u2060-\u2064\u180e]"              # Invisible Characters
)


def sanitize(text):
    """Strip hidden Unicode characters. Returns (cleaned_text, found_chars)."""
    found = HIDDEN_CHAR_PATTERN.findall(text)
    cleaned = HIDDEN_CHAR_PATTERN.sub("", text) if found else text
    return cleaned, found


def format_codepoints(chars, limit=20):
    """Format found characters as U+XXXX strings."""
    return [f"U+{ord(c):04X}" for c in chars[:limit]]
