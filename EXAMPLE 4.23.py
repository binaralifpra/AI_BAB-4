# Example 4.23 The Transformers.py Program
# https://github.com/huggingface/transformers
# pip install transformers
# https://aka.ms/vs/16/release/vc_redist.x64.exe
from transformers import pipeline

classifier = pipeline('sentiment-analysis')
# print() is needed when running as a script (the book ran it interactively)
print(classifier('This is a good movie.'))
