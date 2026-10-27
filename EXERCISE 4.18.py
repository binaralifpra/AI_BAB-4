# Exercise 4.18
# Modify Example 4.22 and choose a different piece of text for text
# analysis, and comment on the results.
import nltk
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

# a text with many names, places and organizations
sentence = """
Barack Obama was born in Hawaii and served as the President of the United States.
In 2004 he gave a speech in Boston, and he later worked with Hillary Clinton.
Google and Microsoft are large technology companies based in California and Washington.
Joko Widodo visited Jakarta and met students from Universitas Indonesia.
"""

tokens = nltk.word_tokenize(sentence)
print(tokens)

tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)

# list only the named entities that were found
print("\nNamed entities:")
for subtree in entities:
    if hasattr(subtree, 'label'):
        name = " ".join(word for word, tag in subtree.leaves())
        print("%-12s %s" % (subtree.label(), name))

# Comment:
# This text has many proper nouns, so ne_chunk finds far more entities than
# in Example 4.22 (where it wrongly marked "Artificial" as a GPE). Most are
# correct: Hawaii, United States, Boston, California, Jakarta are GPE,
# Hillary Clinton and Joko Widodo are PERSON, Microsoft and Universitas
# Indonesia are ORGANIZATION. There are still mistakes: "Barack" and "Obama"
# are split into two PERSON chunks, Google is labelled PERSON instead of
# ORGANIZATION, and Washington (the state) is just GPE so it cannot tell it
# apart from the city. The POS tags are mostly right (names = NNP, "born" =
# VBN). NLTK's chunker is an old statistical model that relies heavily on
# capital letters, so it works on simple English text but makes errors
# that newer transformer models usually avoid.
