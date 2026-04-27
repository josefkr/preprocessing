#!/usr/bin/env python3
"""
Demo script showing how to use the new Stanza_Preprocessor.
"""

from preprocessing.stanza import Stanza_Preprocessor
from preprocessing.api import T_TOKEN, T_SENT, T_POS, T_LEMMA, T_DEP, T_MORPH

# Create a Stanza preprocessor for English
preprocessor = Stanza_Preprocessor(language='en')

# Process text
text = "Dogs love playing. Cats are independent."
cas = preprocessor.run(text)

cas.to_xmi("stanz_demo.xmi")

print(f"Input text: {text}")
print(f"Cleaned text: {cas.sofa_string}")
print()

# Extract and display sentences
print("Sentences:")
for sent in cas.select(T_SENT):
    print(f"  [{sent.begin}:{sent.end}] {sent.get_covered_text()}")
print()

# Extract and display tokens
print("Tokens:")
for token in cas.select(T_TOKEN):
    print(f"  [{token.begin}:{token.end}] {token.get_covered_text()}")
print()

# Extract and display POS tags
print("POS Tags:")
token_pos_map = {}
for token in cas.select(T_TOKEN):
    for pos in cas.select(T_POS):
        if pos.begin == token.begin and pos.end == token.end:
            token_pos_map[token.get_covered_text()] = pos.PosValue
            
for word, tag in token_pos_map.items():
    print(f"  {word}: {tag}")
print()

# Extract and display lemmas
print("Lemmas:")
token_lemma_map = {}
for token in cas.select(T_TOKEN):
    for lemma in cas.select(T_LEMMA):
        if lemma.begin == token.begin and lemma.end == token.end:
            token_lemma_map[token.get_covered_text()] = lemma.value
            
for word, lemma in token_lemma_map.items():
    print(f"  {word} -> {lemma}")
print()

# Extract and display dependencies
print("Dependencies:")
for dep in cas.select(T_DEP):
    gov_text = dep.Governor.get_covered_text()
    dep_text = dep.Dependent.get_covered_text()
    dep_type = dep.DependencyType
    print(f"  {dep_type}({gov_text}, {dep_text})")
