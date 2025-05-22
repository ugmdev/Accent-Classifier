"""
Accent label mapping and formatting utilities.
"""

def get_accent_name(raw_accent):
    """Convert a raw accent label to a human-readable name."""
    mapping = {
        "LABEL_0": "American English",
        "LABEL_1": "British English",
        "LABEL_2": "Indian English",
        "LABEL_3": "Australian English",
        "LABEL_4": "Canadian English",
        "LABEL_5": "Non-native English",
        # Add more mappings as needed
    }
    return mapping.get(raw_accent, raw_accent)

def format_all_results(results):
    """Format a list of accent results for display."""
    return [(get_accent_name(label), prob) for label, prob in results]

class AccentLabelMapper:
    """Class for managing accent label mappings."""
    
    def __init__(self, original_id2label):
        """Initialize with the original ID to label mapping."""
        self.original_id2label = original_id2label
        self.mapped_id2label = self.create_mapped_labels()
    
    def create_mapped_labels(self):
        """Create a mapping from IDs to human-readable accent names."""
        return {idx: get_accent_name(label) for idx, label in self.original_id2label.items()}
