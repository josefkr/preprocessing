import pytest
from cassis import Cas
from preprocessing.api import T_TOKEN, T_SENT, T_POS, T_DEP, T_LEMMA, T_MORPH
from preprocessing.util import get_aslan_typesystem
from preprocessing.stanza import Stanza_Preprocessor


def test_stanza_preprocessing():
    """Test basic Stanza preprocessing for English."""
    text = 'This is a test. A small one.'
    ts = get_aslan_typesystem()
    stanza = Stanza_Preprocessor(language='en')
    cas = stanza.run(text)

    # Test that CAS is created
    assert cas is not None
    assert cas.sofa_string == text
    
    # Test sentence segmentation
    sentences = list(cas.select(T_SENT))
    assert len(sentences) == 2, f"Expected 2 sentences, got {len(sentences)}"
    
    # Test tokens are created
    tokens = list(cas.select(T_TOKEN))
    assert len(tokens) > 0, "Expected tokens to be created"
    
    # Test token text matches
    token_texts = [token.get_covered_text() for token in tokens]
    expected_tokens = ['This', 'is', 'a', 'test', '.', 'A', 'small', 'one', '.']
    assert token_texts == expected_tokens, f"Expected {expected_tokens}, got {token_texts}"
    
    # Test POS annotations
    pos_annotations = list(cas.select(T_POS))
    assert len(pos_annotations) > 0, "Expected POS annotations"
    
    # Test that lemmas are created
    lemma_annotations = list(cas.select(T_LEMMA))
    assert len(lemma_annotations) > 0, "Expected lemma annotations"
    
    # Test dependencies
    dependencies = list(cas.select(T_DEP))
    assert len(dependencies) > 0, "Expected dependency annotations"


def test_stanza_pos_and_lemmas():
    """Test that POS tags and lemmas are correctly extracted.

    DKPro convention: ``PosValue`` holds the fine-grained / Penn-Treebank
    XPOS, ``coarseValue`` holds the UD UPOS.
    """
    text = 'The dogs are running.'
    stanza = Stanza_Preprocessor(language='en')
    cas = stanza.run(text)

    # Get tokens and their POS tags
    upos_dict = {}
    xpos_dict = {}
    lemma_dict = {}

    for token in cas.select(T_TOKEN):
        token_text = token.get_covered_text()
        token_begin = token.begin

        for pos in cas.select(T_POS):
            if pos.begin == token_begin:
                upos_dict[token_text] = pos.coarseValue
                xpos_dict[token_text] = pos.PosValue
                break

        for lemma in cas.select(T_LEMMA):
            if lemma.begin == token_begin:
                lemma_dict[token_text] = lemma.value
                break

    # UD POS (coarseValue)
    assert upos_dict.get('The') in ['DET', 'PRON'], f"Expected DET or PRON for 'The', got {upos_dict.get('The')}"
    assert upos_dict.get('dogs') == 'NOUN', f"Expected NOUN for 'dogs', got {upos_dict.get('dogs')}"
    assert upos_dict.get('are') in ['AUX', 'VERB'], f"Expected AUX or VERB for 'are', got {upos_dict.get('are')}"
    assert upos_dict.get('running') == 'VERB', f"Expected VERB for 'running', got {upos_dict.get('running')}"

    # Penn-Treebank XPOS (PosValue)
    assert xpos_dict.get('The') == 'DT', f"Expected DT for 'The', got {xpos_dict.get('The')}"
    assert xpos_dict.get('dogs') in ['NNS', 'NN'], f"Expected NNS for 'dogs', got {xpos_dict.get('dogs')}"

    # Lemmas
    assert lemma_dict.get('dogs') == 'dog', f"Expected 'dog' for lemma of 'dogs', got {lemma_dict.get('dogs')}"
    assert lemma_dict.get('running') == 'run', f"Expected 'run' for lemma of 'running', got {lemma_dict.get('running')}"


def test_stanza_dependencies():
    """Test that dependency relations are correctly extracted."""
    text = 'Dogs run.'
    stanza = Stanza_Preprocessor(language='en')
    cas = stanza.run(text)
    
    dependencies = list(cas.select(T_DEP))
    assert len(dependencies) > 0, "Expected at least one dependency"
    
    # Find the root dependency
    root_deps = [d for d in dependencies if d.DependencyType in ['ROOT', 'root']]
    assert len(root_deps) > 0, "Expected a ROOT dependency"
    
    # Check that dependencies have governor and dependent
    for dep in dependencies:
        assert dep.Governor is not None, "Governor should not be None"
        assert dep.Dependent is not None, "Dependent should not be None"
        assert dep.DependencyType is not None, "DependencyType should not be None"


def test_stanza_lazy_loading():
    """Test that Stanza pipelines are loaded lazily and cached."""
    from preprocessing.stanza import _STANZA_CACHE
    
    # Clear cache before test
    _STANZA_CACHE.clear()
    
    # 1. Test that pipeline is not loaded on instantiation
    stanza_preprocessor = Stanza_Preprocessor(language='en')
    assert stanza_preprocessor.pipeline is None, "Pipeline should not be loaded on instantiation"
    
    # 2. Test that pipeline is loaded on first use
    text = "This is a test."
    cas = stanza_preprocessor.run(text)
    assert stanza_preprocessor.pipeline is not None, "Pipeline should be loaded after first use"
    first_pipeline = stanza_preprocessor.pipeline
    
    # 3. Test that cache is used for subsequent instantiations
    stanza_preprocessor2 = Stanza_Preprocessor(language='en')
    assert stanza_preprocessor2.pipeline is None, "Second instance should also have lazy loading"
    
    cas2 = stanza_preprocessor2.run(text)
    assert stanza_preprocessor2.pipeline is first_pipeline, "Second instance should use cached pipeline"
    
    # 4. Test cache contains the pipeline
    assert 'en' in _STANZA_CACHE, "Pipeline should be in cache"
    assert _STANZA_CACHE['en'] is first_pipeline, "Cached pipeline should be the same instance"


def test_stanza_text_cleaning():
    """Test that text cleaning works correctly."""
    # Text with control characters and extra whitespace
    text = "Hello  \t  world \x00 test"
    stanza = Stanza_Preprocessor(language='en')
    cas = stanza.run(text)
    
    # Control characters should be removed and whitespace normalized
    expected = "Hello world test"
    assert cas.sofa_string == expected, f"Expected '{expected}', got '{cas.sofa_string}'"
