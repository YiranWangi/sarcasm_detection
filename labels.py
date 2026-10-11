"""Labels shared by training, generation and evaluation."""

ID_TO_LABEL = {0: "非反讽", 1: "反讽"}
LABEL_TO_ID = {label: index for index, label in ID_TO_LABEL.items()}
UNKNOWN_LABEL = "未知"


def parse_label(text):
    """Return an integer label, or None for an invalid generated answer."""
    return LABEL_TO_ID.get(text.strip()) if isinstance(text, str) else None
