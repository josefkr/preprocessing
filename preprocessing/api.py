import unicodedata
from abc import ABC, abstractmethod

from cassis import Cas

T_ANOMALY = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.SpellingAnomaly"
T_SUGGESTION = "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.SuggestedAction"
T_SENT = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
T_TOKEN = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
T_LEMMA = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
T_DEP = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
T_MORPH = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.Morpheme"
T_POS = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"
T_NER = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"


class BasePreprocessor(ABC):
    """Abstract base class for text preprocessors.

    Provides common functionality such as typesystem management and text cleaning.
    Subclasses must implement the run() method to define language-specific processing.
    """

    def __init__(self, language: str):
        """Initialize the preprocessor.

        Args:
            language: Language code (e.g., 'en', 'de', 'fr')
        """
        from preprocessing.util import get_aslan_typesystem

        self.language = language
        self.ts = get_aslan_typesystem()

    def _clean_string(self, text: str) -> str:
        """Remove control characters and extra whitespace.

        Args:
            text: Input text to clean

        Returns:
            Cleaned text with control characters removed and whitespace normalized
        """
        cleaned = "".join(ch for ch in text if unicodedata.category(ch) != "Cc")
        unstretched = " ".join(cleaned.split())
        return unstretched

    @abstractmethod
    def run(self, text: str) -> Cas:
        """Process text and return a CASSIS CAS object with annotations.

        Args:
            text: Input text to process

        Returns:
            Cas: CASSIS CAS object with extracted annotations
        """
        pass
