"""Put untrusted filing text into a prompt so it stays data, never instructions."""

import re

_TAG_LOOKALIKE = re.compile(r"<(/?)filing_text", re.IGNORECASE)


def filing_text_block(text: str, **attributes: str) -> str:
    """Wrap text in <filing_text ...> tags, defusing any tag look-alikes inside it.

    Filing text is outside text: if it contained a closing tag, everything after it would look
    like it came from us. Escaping the look-alikes means the data can't leave its box.
    """
    safe = _TAG_LOOKALIKE.sub(r"&lt;\1filing_text", text)
    attrs = "".join(f' {name}="{value}"' for name, value in attributes.items())
    return f"<filing_text{attrs}>\n{safe}\n</filing_text>"
