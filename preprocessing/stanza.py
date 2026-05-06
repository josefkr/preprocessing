import logging
import stanza
from cassis import Cas

from preprocessing.api import BasePreprocessor, T_TOKEN, T_SENT, T_POS, T_DEP, T_LEMMA, T_MORPH

logger = logging.getLogger(__name__)

# Default language for Stanza pipeline
DEFAULT_LANGUAGE = "en"

# Cache loaded models to avoid reloading
_STANZA_CACHE = {}


class Stanza_Preprocessor(BasePreprocessor):
    def __init__(self, language: str = DEFAULT_LANGUAGE):
        """
        Initialize the Stanza preprocessor.
        
        Args:
            language: Language code (e.g., 'en', 'de', 'fr')
                      Passed directly to stanza.Pipeline()
        """
        super().__init__(language)
        
        # Lazy load pipeline on first use
        self.pipeline = None
    
    def _load_pipeline(self):
        """Lazy load the Stanza pipeline on first use."""
        if self.pipeline is not None:
            return self.pipeline
        
        # Check cache first
        if self.language in _STANZA_CACHE:
            self.pipeline = _STANZA_CACHE[self.language]
            return self.pipeline
        
        try:
            logger.info(f"Loading Stanza pipeline for language: {self.language}")
            # Initialize pipeline with tokenization, POS tagging, lemmatization, dependency parsing
            # The 'mwt' processor expands multi-word tokens
            self.pipeline = stanza.Pipeline(
                lang=self.language,
                processors="tokenize,mwt,pos,lemma,depparse",
                verbose=False,
                logging_level=logging.WARNING,
            )
            _STANZA_CACHE[self.language] = self.pipeline
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Stanza pipeline for language '{self.language}'. "
                f"Ensure stanza is installed and the language model is available.\n"
                f"Download with: python -m stanza.download('{self.language}')"
            ) from e
        
        return self.pipeline

    def run(self, text: str) -> Cas:
        """
        Process text and return a CASSIS CAS object with annotations.
        
        Args:
            text: Input text to process
            
        Returns:
            Cas: CASSIS CAS object with extracted annotations
        """
        self.cas = Cas(self.ts)
        
        # Clean and normalize text
        cleaned = self._clean_string(text)
        self.cas.sofa_string = cleaned
        
        # Load pipeline and process text
        pipeline = self._load_pipeline()
        doc = pipeline(cleaned)
        
        # Get type handles for annotations
        T = self.ts.get_type(T_TOKEN)
        S = self.ts.get_type(T_SENT)
        P = self.ts.get_type(T_POS)
        D = self.ts.get_type(T_DEP)
        L = self.ts.get_type(T_LEMMA)
        M = self.ts.get_type(T_MORPH)
        
        # First pass: add sentences
        for sentence in doc.sentences:
            # Get character offsets from first and last word
            first_word = sentence.tokens[0].words[0]
            last_word = sentence.tokens[-1].words[-1]
            
            cas_sentence = S(
                begin=first_word.start_char,
                end=last_word.end_char
            )
            self.cas.add(cas_sentence)
        
        # Second pass: add tokens and their annotations
        token_annos = []  # Track token annotations for dependency linking
        token_map = {}    # Map from (sentence_idx, token_idx, word_idx) to annotation
        
        global_token_id = 0
        
        for sent_idx, sentence in enumerate(doc.sentences):
            for token_idx, token in enumerate(sentence.tokens):
                # Each token can have multiple words (multiword tokens)
                # We expand them into separate CAS tokens
                for word_idx, word in enumerate(token.words):
                    # Extract character offsets
                    begin = word.start_char
                    end = word.end_char
                    
                    # Create POS annotation. DKPro convention:
                    #   PosValue    = fine-grained (XPOS, e.g. Penn Treebank)
                    #   coarseValue = coarse UD POS (UPOS)
                    cas_pos = P(
                        begin=begin,
                        end=end,
                        PosValue=word.xpos or "",
                        coarseValue=word.upos or "",
                    )
                    self.cas.add(cas_pos)
                    
                    # Create lemma annotation
                    lemma_value = word.lemma or ""
                    cas_lemma = L(
                        begin=begin,
                        end=end,
                        value=lemma_value
                    )
                    self.cas.add(cas_lemma)
                    
                    # Create morphology annotation if features exist
                    if word.feats:
                        # Feats is already a string in this format: "Case=Nom|Number=Sing"
                        cas_morph = M(
                            begin=begin,
                            end=end,
                            morphTag=word.feats
                        )
                        self.cas.add(cas_morph)
                    
                    # Create token annotation
                    cas_token = T(
                        begin=begin,
                        end=end,
                        id=global_token_id,
                        pos=cas_pos,
                        lemma=cas_lemma
                    )
                    self.cas.add(cas_token)
                    
                    token_annos.append(cas_token)
                    token_map[(sent_idx, token_idx, word_idx)] = (cas_token, word)
                    global_token_id += 1
        
        # Third pass: add dependency relations
        # Need all tokens in CAS first
        for sent_idx, sentence in enumerate(doc.sentences):
            for token_idx, token in enumerate(sentence.tokens):
                for word_idx, word in enumerate(token.words):
                    dependent_anno, dependent_word = token_map[(sent_idx, token_idx, word_idx)]
                    
                    # Get the head token
                    head_idx = word.head
                    
                    if head_idx == 0:
                        # Root token: self-relation
                        governor_anno = dependent_anno
                    else:
                        # Regular dependency - head_idx points to a word in the sentence
                        # We need to find which token/word pair matches this word index
                        # Stanza uses 1-based head indices within a sentence
                        word_count = 0
                        governor_anno = None
                        
                        for t_idx, t in enumerate(sentence.tokens):
                            for w_idx, w in enumerate(t.words):
                                word_count += 1
                                if word_count == head_idx:
                                    if (sent_idx, t_idx, w_idx) in token_map:
                                        governor_anno, _ = token_map[(sent_idx, t_idx, w_idx)]
                                    break
                            if governor_anno is not None:
                                break
                        
                        if governor_anno is None:
                            logger.warning(
                                f"Could not find head token with index {head_idx} "
                                f"for word '{dependent_word.text}' in sentence {sent_idx}"
                            )
                            continue
                    
                    # Create dependency annotation
                    dep_type = word.deprel or "dep"
                    cas_dep = D(
                        begin=dependent_anno.begin,
                        end=dependent_anno.end,
                        Governor=governor_anno,
                        Dependent=dependent_anno,
                        DependencyType=dep_type,
                        flavor="basic"
                    )
                    self.cas.add(cas_dep)
        
        return self.cas
