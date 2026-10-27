# Example 4.24 The Transformers2.py Program
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

print(question_answerer({
    'question': 'What is the name of the company?',
    'context': 'We created Biox Systems Ltd company back in the year of 2000.'
}))
