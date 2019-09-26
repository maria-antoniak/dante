import argparse
from collections import defaultdict
from datetime import datetime
import gensim
import json
from operator import itemgetter
import os
import pandas as pd
import pickle
import pyLDAvis
import re


STOPS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll', 'amp',
         'thou', 'thine', 'thy', 'doth', 'art', 'hath', 'thee', 'er', 'dost', 'st']


def remove_extra_spaces(text):
    return ' '.join(text.split())


def remove_non_alpha(text):
    # text = re.sub('[0-9]+', 'NUM', text)
    return re.sub('[^A-Za-z\s]', ' ', text)


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in STOPS])


def remove_short_words(text, min_length):
    return ' '.join([word for word in text.split() if not len(word) < min_length])


def process_text(text):
    text = text.lower()
    text = remove_non_alpha(text)
    text = remove_stop_words(text)
    text = remove_short_words(text, 2)
    text = remove_extra_spaces(text)
    return text


def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def get_training_data(texts):

    original_texts = []
    training_documents = []
    text_ids = []
    times = []

    text_lengths_dict = defaultdict(int)

    for i, _lines in enumerate(texts):

        t = 0   # Keep this counter instead of using enumerate() because not all paragraphs are used.

        for _line in _lines:

            processed_unigrams = process_text(_line).split()

            if len(processed_unigrams) >= 4:

                original_texts.append(_line)
                training_documents.append(processed_unigrams)
                text_ids.append(i)

                times.append(t)
                text_lengths_dict[i] += 1
                t += 1

    # Normalize the paragraph times by the total number of paragraphs in each story.
    times = [time/float(text_lengths_dict[text_id]) for time, text_id in zip(times, text_ids)]
        
    return original_texts, training_documents, text_ids, times


def get_training_data_cantos(texts):

    original_texts = []
    training_documents = []
    text_ids = []
    times = []

    text_lengths_dict = defaultdict(int)

    for i, _lines in enumerate(texts):

        canto_index = 0   # Keep this counter instead of using enumerate() because not all paragraphs are used.

        for _line in _lines:

            if 'canto' in _line.lower().split():
                # Beginning of new canto!
                canto_index += 1
                original_texts.append('')
                training_documents.append([])
                text_ids.append(i)
                times.append(canto_index)

            else:
                processed_unigrams = process_text(_line).split()
                original_texts[-1] += _line + '\n'
                training_documents[-1] += processed_unigrams
        
    return original_texts, training_documents, text_ids, times


def load_model(gensim_dictionary_path, gensim_corpus_path, gensim_model_path):
    """
    Loads the gensim LDA model object and accompanying dictionary and corpus objects.
    """

    model = gensim.models.LdaModel.load(gensim_model_path)
    dictionary = gensim.corpora.Dictionary.load(gensim_dictionary_path)
    corpus = gensim.corpora.MmCorpus(gensim_corpus_path)

    return dictionary, corpus, model


def display_topics(model, dictionary, num_topics, output_path):
    """
    Gets the top N words for each topic and saves the results to an output file.
    """

    # topics = model.show_topics(num_topics=num_topics)

    output_file = open(output_path, 'w')

    # for i in range(0, num_topics):

        # word_probability_tuples = model.get_topic_terms(i, topn=20)

        # _words = [dictionary[word_id] for word_id, probability in _word_probability_tuples]
        # output_file.write(' '.join(_words) + '\n')
        # print(_words)

    topics = model.print_topics(num_topics=-1, num_words=10)

    for _topic in topics:
        _words = [t.split('*')[1].strip().strip('\"') for t in _topic[1].split('+')]

        output_file.write(' '.join(_words) + '\n')
        print(_words)


def run_pyLDAvis(dictionary, model, corpus):
    """
    Serves pyLDAvis to interactively explore the topics.
    """
    topics_data = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.show(topics_data, port=8889)


def get_topic_assignments(model):

    topic_assignments = []

    topic_distributions = model.load_document_topics()

    for topic_prob_tuples in topic_distributions:
    
        _highest_prob = max(topic_prob_tuples, key=itemgetter(1))[1]
        _most_probable_topics = [_topic 
                                 for _topic, _prob in topic_prob_tuples 
                                 if _prob == _highest_prob]
        
        topic_assignments.append(_most_probable_topics)

    return topic_assignments


def get_topic_distributions(model):

    topic_distributions = []

    topic_distribution_generator = model.load_document_topics()

    for topic_prob_tuples in topic_distribution_generator:
        topic_distributions.append(topic_prob_tuples)

    return topic_distributions


def train_and_save_model(mallet_path,
                         documents, 
                         gensim_dict_path, 
                         gensim_corpus_path, 
                         gensim_model_path, 
                         num_topics):

    print('-- Building dictionary...')
    dictionary = gensim.corpora.Dictionary(documents)
    dictionary.compactify()
    dictionary.save(gensim_dict_path)

    print('-- Building corpus...')
    corpus = [dictionary.doc2bow(document) for document in documents]
    gensim.corpora.MmCorpus.serialize(gensim_corpus_path, corpus)

    print('-- Training...')
    model = gensim.models.wrappers.LdaMallet(mallet_path,
                                             corpus=corpus,
                                             num_topics=num_topics,
                                             id2word=dictionary)
    gensim.models.wrappers.LdaMallet.save(model, gensim_model_path)

    return dictionary, corpus, model


def read_texts(directory_path, file_names):

    texts = []

    for _file_name in file_names:
        
        with open(directory_path + '/' + _file_name, 
                'r', 
                encoding='utf-8', 
                errors='ignore') as _book_file:
        
            _lines = _book_file.read().split('\n')
            _lines = [_line.strip() for _line in _lines if _line.strip()]
            texts.append(_lines)
    
    return texts


def main():

    start_time = datetime.now()

    mallet_path           = '/Users/mah343/Documents/packages/mallet-2.0.8/bin/mallet'
    output_directory_path = '/Volumes/MARIA/output/translations/dante/topics'
    input_directory_path  = '/Volumes/MARIA/datasets/dante'
    input_file_names = ['longfellow_inferno_gutenberg.txt', 'longfellow_purgatario_gutenberg.txt', 'longfellow_paradiso_gutenberg.txt',
                        'cary_inferno_gutenberg.txt', 'cary_purgatario_gutenberg.txt', 'cary_paradiso_gutenberg.txt']

    label = 'cantos' # 'lines', 'stanzas', 'cantos' ... ?
    num_topics = 50

    json_path = output_directory_path + '/' + str(num_topics) + '.'  + label + '.training_data.json'

    gensim_dictionary_path = output_directory_path + '/' + str(num_topics) + '.' + label + '.gensim.dict'
    gensim_corpus_path = output_directory_path + '/' + str(num_topics) + '.' + label + '.gensim.corpus'
    gensim_model_path = output_directory_path + '/' + str(num_topics) + '.' + label + '.gensim.model'
    topics_output_path = output_directory_path + '/' + str(num_topics) + '.' + label + '.topics.txt'

    texts = read_texts(input_directory_path, input_file_names)

    print('Number of texts: ' + str(len(texts)))
    print('Number of lines in texts: ' + str([len(t) for t in texts]))

    # Process and filter the training data.
    if label == 'lines':
        original_texts, training_documents, text_ids, times = get_training_data(texts)
    elif label == 'cantos':
        original_texts, training_documents, text_ids, times = get_training_data_cantos(texts)

    # Train and save the model.
    dictionary, corpus, model = train_and_save_model(mallet_path,
                                                     training_documents,
                                                     gensim_dictionary_path,
                                                     gensim_corpus_path,
                                                     gensim_model_path,
                                                     num_topics)

    # Display the results of training.
    display_topics(model, dictionary, num_topics, topics_output_path)

    # Get topic assignments for each training document.
    topic_assignments = get_topic_assignments(model)
    topic_distributions = get_topic_distributions(model)

    # Save all data. 
    json.dump({'original_texts': original_texts,
               'training_documents': training_documents,
               'text_ids': text_ids,
               'times': times,
               'topic_assignments': topic_assignments,
               'topic_distributions': topic_distributions}, open(json_path, 'w'))

    print('-- Run Time = ' + str(datetime.now() - start_time))


if __name__ == '__main__':
    main()