# Exercise 4.20
# Modify Example 4.24, choose a different piece of text, ask different
# questions, and comment on the results.
# Transformers v5 removed the 'question-answering' pipeline, so this does
# the same job by hand with the model the old pipeline used by default.
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = 'distilbert/distilbert-base-cased-distilled-squad'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def question_answerer(inputs):
    question, context = inputs['question'], inputs['context']
    enc = tokenizer(question, context, return_tensors='pt',
                    return_offsets_mapping=True)
    offsets = enc.pop('offset_mapping')[0]
    with torch.no_grad():
        out = model(**enc)
    # only allow the answer to come from the context, not the question
    is_context = torch.tensor([s == 1 for s in enc.sequence_ids(0)])
    start_logits = out.start_logits[0].masked_fill(~is_context, -1e9)
    end_logits = out.end_logits[0].masked_fill(~is_context, -1e9)
    # best (start, end) pair with start <= end
    scores = start_logits.softmax(-1)[:, None] * end_logits.softmax(-1)[None, :]
    scores = torch.triu(scores)
    s, e = divmod(int(scores.argmax()), scores.shape[1])
    start, end = int(offsets[s][0]), int(offsets[e][1])
    return {'answer': context[start:end], 'end': end,
            'score': float(scores[s, e]), 'start': start}

context = """
Artificial intelligence (AI) is intelligence demonstrated by machines.
The term was coined by John McCarthy in 1956 at a workshop held at
Dartmouth College. Deep learning became popular after AlexNet won the
ImageNet competition in 2012. Transformers were introduced by researchers
at Google in 2017 and are mainly used for natural language processing.
"""

questions = [
    'Who coined the term artificial intelligence?',
    'When was the term artificial intelligence coined?',
    'Where was the workshop held?',
    'Which network won the ImageNet competition in 2012?',
    'What are transformers mainly used for?',
    'Who introduced transformers?',
    'What is the capital of France?',
]

for q in questions:
    result = question_answerer({'question': q, 'context': context})
    print("Q: %s" % q)
    print("A: %-40s score: %.4f\n" % (result['answer'], result['score']))

# Comment:
# The model answers every question that has an answer in the text
# correctly: who (John McCarthy), when (1956), where (Dartmouth College),
# which (AlexNet) and what (natural language processing), mostly with
# scores above 0.94. "Who introduced transformers?" gets "Google" with a
# lower score (0.74), because the text says "researchers at Google", so
# the model is less sure which words to pick. The last question has no
# answer in the text, but the model still returns "Dartmouth College"
# (score 0.56). This is extractive question answering: it can only copy a
# piece of the context and always returns something, so a low score is a
# sign that the answer may be wrong.
