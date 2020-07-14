# -*- coding: utf-8 -*-

from models.readers import Reader
from models.joint_bert.vectorizers.bert_vectorizer import BERTVectorizer
from models.joint_bert.models.joint_bert import JointBertModel
from models.joint_bert.vectorizers.tags_vectorizer import TagsVectorizer
from models.joint_bert.joint_bert_config import config

import argparse
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import pickle
import tensorflow as tf


# setting config info
train_data_folder_path = config["train"]
val_data_folder_path = config["val"]
save_folder_path = config["save"]
epochs = config["epochs"]
batch_size = config["batch"]
model_type = config["type"]
start_model_folder_path = config["start_model"]
bert_model_hub_path = config["hub_url"][model_type]
is_bert = True if model_type == 'bert' else False


# tf.compat.v1.random.set_random_seed(7)

print('reading data ...')
train_text_arr, train_tags_arr, train_intents = Reader.read_goo_format(train_data_folder_path)
val_text_arr, val_tags_arr, val_intents = Reader.read_goo_format(val_data_folder_path)

print(is_bert)
print('vectorize data ...')
bert_vectorizer = BERTVectorizer(is_bert, bert_model_hub_path)
train_input_ids, train_input_mask, train_segment_ids, train_valid_positions, train_sequence_lengths = bert_vectorizer.transform(train_text_arr)
val_input_ids, val_input_mask, val_segment_ids, val_valid_positions, val_sequence_lengths = bert_vectorizer.transform(val_text_arr)


print('vectorize tags ...')
tags_vectorizer = TagsVectorizer()
tags_vectorizer.fit(train_tags_arr)
train_tags = tags_vectorizer.transform(train_tags_arr, train_valid_positions)
val_tags = tags_vectorizer.transform(val_tags_arr, val_valid_positions)
slots_num = len(tags_vectorizer.label_encoder.classes_)


print('encode labels ...')
intents_label_encoder = LabelEncoder()
train_intents = intents_label_encoder.fit_transform(train_intents).astype(np.int32)
val_intents = intents_label_encoder.transform(val_intents).astype(np.int32)
intents_num = len(intents_label_encoder.classes_)

if start_model_folder_path is None or start_model_folder_path == '':
    model = JointBertModel(slots_num, intents_num, bert_model_hub_path,
                           num_bert_fine_tune_layers=10, is_bert=is_bert)
else:
    model = JointBertModel.load(start_model_folder_path)

print('training model ...')
model.fit([train_input_ids, train_input_mask, train_segment_ids, train_valid_positions], [train_tags, train_intents],
          validation_data=([val_input_ids, val_input_mask, val_segment_ids, val_valid_positions], [val_tags, val_intents]),
          epochs=epochs, batch_size=batch_size)

# print(model.predict("add leah kauffman to my uncharted 4 nathan drake playlist"))

# saving
print('Saving ..')
if not os.path.exists(save_folder_path):
    os.makedirs(save_folder_path)
    print('Folder `%s` created' % save_folder_path)
model.save(save_folder_path)
with open(os.path.join(save_folder_path, 'tags_vectorizer.pkl'), 'wb') as handle:
    pickle.dump(tags_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(save_folder_path, 'intents_label_encoder.pkl'), 'wb') as handle:
    pickle.dump(intents_label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)


tf.compat.v1.reset_default_graph()