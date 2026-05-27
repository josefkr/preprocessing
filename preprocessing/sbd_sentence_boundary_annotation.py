"""Sentence segmentation implementation using pysbd."""

import pysbd

from aslan_normalization.normalization import Normalizer


class SentenceSegmentationNormalizer(Normalizer):
    """Segment text into sentences using pysbd.

    This normalizer splits the input text into individual sentences
    and joins them with newlines. It uses pysbd (pragma sentence
    boundary disambiguation) for robust sentence boundary detection.
    """

    OUTPUT_VIEW_NAME = "pysbd_segmented"

    def __init__(self, language: str = "de", clean: bool = False) -> None:
        """Initialize the sentence segmenter.

        Args:
            language: Language code for pysbd (default: "de" for German).
            clean: Whether pysbd should clean the output (default: False).
        """
        self.language = language
        self.clean = clean

    def transform(self, text: str) -> str:
        """Segment text into sentences separated by newlines.

        Args:
            text: The input text to segment.

        Returns:
            Text with sentences separated by newlines.
        """
        segmenter = pysbd.Segmenter(language=self.language, clean=self.clean)
        sentences = segmenter.segment(text)
        return "\n".join(sentences)
