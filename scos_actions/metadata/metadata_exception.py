class MetadataException(Exception):
    """Basic exception handling for metadata-related problems."""

    def __init__(self, msg):
        super().__init__(msg)
