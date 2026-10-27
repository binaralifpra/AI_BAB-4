# Exercise 4.19
# Modify Example 4.23, choose some different sentences, make sure some are
# positive and some are negative, and comment on the results.
from transformers import pipeline

classifier = pipeline('sentiment-analysis')

sentences = [
    'This is a good movie.',
    'I really enjoyed the food, the service was excellent!',
    'The lecture today was clear and very helpful.',
    'This is a bad movie.',
    'The phone stopped working after two days, what a waste of money.',
    'I hate waiting in long queues.',
    'The movie was not bad at all.',
    'Oh great, my laptop crashed again right before the deadline.',
    'The meeting is at 10 am.',
]

for text, result in zip(sentences, classifier(sentences)):
    print("%-9s %.4f  %s" % (result['label'], result['score'], text))

# Comment:
# The three clearly positive and three clearly negative sentences are all
# classified correctly with scores above 0.99. The model also handles the
# harder cases: "not bad at all" is POSITIVE (it understands the negation)
# and the sarcastic "Oh great, my laptop crashed..." is NEGATIVE even though
# it contains the word "great", so it looks at the whole sentence, not just
# single words. Its weak point is neutral text: "The meeting is at 10 am."
# has no sentiment but is still labelled POSITIVE (0.96), because the model
# only has two classes and must always pick one. A high score therefore
# does not always mean the answer makes sense.
