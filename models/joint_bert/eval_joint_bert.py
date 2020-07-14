# -*- coding: utf-8 -*-
from models.readers import Reader
from models.joint_bert.vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert.models.joint_bert import JointBertModel
from models.joint_bert.utils import flatten

import argparse
import os
import pickle
import tensorflow as tf
from sklearn import metrics
from models.joint_bert.joint_bert_config import config

load_folder_path = config["save"]
data_folder_path = config["test"]
batch_size = config["batch"]
model_type = config["type"]
bert_model_hub_path = config["hub_url"][model_type]
is_bert = True if model_type == 'bert' else False

bert_vectorizer = BERTVectorizer(is_bert, bert_model_hub_path)

# loading models
print('Loading models ...')
if not os.path.exists(load_folder_path):
    print('Folder `%s` not exist' % load_folder_path)

with open(os.path.join(load_folder_path, 'tags_vectorizer.pkl'), 'rb') as handle:
    tags_vectorizer = pickle.load(handle)
    slots_num = len(tags_vectorizer.label_encoder.classes_)
with open(os.path.join(load_folder_path, 'intents_label_encoder.pkl'), 'rb') as handle:
    intents_label_encoder = pickle.load(handle)
    intents_num = len(intents_label_encoder.classes_)

model = JointBertModel.load(load_folder_path, )

data_text_arr, data_tags_arr, data_intents = Reader.read_goo_format(data_folder_path)
data_input_ids, data_input_mask, data_segment_ids, data_valid_positions, data_sequence_lengths = bert_vectorizer.transform(
    data_text_arr)


def get_results(input_ids, input_mask, segment_ids, valid_positions, sequence_lengths, tags_arr,
                intents, tags_vectorizer, intents_label_encoder):
    predicted_tags, predicted_intents = model.predict_slots_intent(
        [input_ids, input_mask, segment_ids, valid_positions],
        tags_vectorizer, intents_label_encoder, remove_start_end=True)
    gold_tags = [x.split() for x in tags_arr]
    # print(metrics.classification_report(flatten(gold_tags), flatten(predicted_tags), digits=3))
    f1_score = metrics.f1_score(flatten(gold_tags), flatten(predicted_tags), average='micro')
    acc = metrics.accuracy_score(intents, predicted_intents)
    return f1_score, acc


print('==== Evaluation ====')
f1_score, acc = get_results(data_input_ids, data_input_mask, data_segment_ids, data_valid_positions,
                            data_sequence_lengths,
                            data_tags_arr, data_intents, tags_vectorizer, intents_label_encoder)
print('Slot f1_score = %f' % f1_score)
print('Intent accuracy = %f' % acc)

tf.compat.v1.reset_default_graph()
