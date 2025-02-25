'''MIT License

Copyright (c) 2025 Prateek De

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
THIS IS A PROPERTY OF MINDSCRIBE TECH.. PLEASE DO NOT COPY OR DISTRIBUTE WITHOUT PERMISSION
'''



# Example using NLTK for extractive summarization
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

def summarize(text, num_sentences=2):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    freq_table = {}
    for word in words:
        word = word.lower()
        if word not in stop_words and word.isalnum():
            freq_table[word] = freq_table.get(word, 0) + 1
    sentences = sent_tokenize(text)
    sentence_scores = {}
    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary_sentences = ranked_sentences[:num_sentences]
    summary = ' '.join(summary_sentences)
    return summary

# Example usage
text = """
Artificial Intelligence (AI) has significantly transformed various industries over the past decade. One of the most important applications of AI is Natural Language Processing (NLP), which enables machines to understand, interpret, and generate human language. NLP is widely used in applications such as chatbots, machine translation, speech recognition, and sentiment analysis.

Text summarization is a crucial NLP task that involves condensing a large text into a concise version while preserving essential information. Summarization techniques are broadly categorized into extractive and abstractive methods. Extractive summarization selects key sentences from the original text and presents them as a summary, whereas abstractive summarization generates new sentences that convey the main ideas in a human-like manner.

Various algorithms power extractive summarization. One of the simplest techniques is the frequency-based method, where words are assigned scores based on their occurrence in the text, and sentences containing the most frequent words are included in the summary. More advanced methods, such as TF-IDF (Term Frequency-Inverse Document Frequency), improve summarization by considering the importance of words relative to the entire document.

In contrast, abstractive summarization relies on deep learning models such as transformers and recurrent neural networks. These models generate summaries that capture the essence of a document while paraphrasing content naturally. Transformers, particularly those based on architectures like BERT and GPT, have revolutionized NLP by achieving human-level summarization accuracy.

As AI and NLP continue to evolve, text summarization will play a vital role in information retrieval, allowing users to process large amounts of data efficiently. Whether used in news aggregation, legal document analysis, or academic research, summarization helps individuals and businesses make informed decisions in a fast-paced world.

"""
summary = summarize(text)
print(summary)
