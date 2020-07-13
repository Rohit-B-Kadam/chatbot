# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class BERTVectorizer:

    def __init__(self, is_bert, bert_model_hub_path):
        self.is_bert = is_bert
        self.bert_model_hub_path = bert_model_hub_path
        self.create_tokenizer_from_hub_module(is_bert=is_bert)

    def create_tokenizer_from_hub_module(self, is_bert):
        """Get the vocab file and casing info from the Hub module."""
        # bert_module =  hub.Module(self.bert_model_hub_path)
        module_layer = hub.KerasLayer(self.bert_model_hub_path,
                                      trainable=False)

        if is_bert:
            from models.joint_bert.vectorizers.tokenization import FullTokenizer
            vocab_file = module_layer.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = module_layer.resolved_object.do_lower_case.numpy()
            self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        else:
            sp_model_file = module_layer.resolved_object.sp_model_file.asset_path.numpy()

            # commented and used the below instead because of lower case problem
            # from vectorizers.tokenization import FullSentencePieceTokenizer
            # self.tokenizer = FullSentencePieceTokenizer(sp_model_file)
            from models.joint_bert.vectorizers.albert_tokenization import FullTokenizer
            self.tokenizer = FullTokenizer(vocab_file=sp_model_file,
                                           do_lower_case=True,
                                           spm_model_file=sp_model_file)

        del module_layer

    def tokenize(self, text: str):
        words = text.split()  # whitespace tokenizer
        # text: add leah kauffman to my uncharted 4 nathan drake playlist
        # words: ['add', 'leah', 'kauffman', 'to', 'my', 'uncharted', '4', 'nathan', 'drake', 'playlist']

        tokens = []
        valid_positions = []

        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)

        # tokens: ['add', 'leah', 'ka', '##uf', '##fm', '##an', 'to', 'my', 'un', '##cha', '##rted', '4', 'nathan', 'drake', 'play', '##list']
        # valid_positions:[1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
        return tokens, valid_positions

    def __vectorize(self, text: str):
        tokens, valid_positions = self.tokenize(text)
        # insert the first "[CLS]"
        tokens.insert(0, '[CLS]')
        valid_positions.insert(0, 1)
        # insert the last token "[SEP]"
        tokens.append('[SEP]')
        valid_positions.append(1)
        # ['[CLS]', 'add', 'leah', 'ka', '##uf', '##fm', '##an', 'to', 'my', 'un', '##cha', '##rted', '4', 'nathan', 'drake', 'play', '##list', '[SEP]']
        # [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1]

        segment_ids = [0] * len(tokens)
        # segment_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        # input_ids: [101, 5587, 14188, 10556, 16093, 16715, 2319, 2000, 2026, 4895, 7507, 17724, 1018, 7150, 7867, 2377, 9863, 102] and the first is always 101 and the last is 102
        input_mask = [1] * len(input_ids)
        # input_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return input_ids, input_mask, segment_ids, valid_positions

    def transform(self, text_arr):
        input_ids = []
        input_mask = []
        segment_ids = []
        valid_positions = []
        for text in text_arr:
            ids, mask, seg_ids, valid_pos = self.__vectorize(text)
            input_ids.append(ids)
            input_mask.append(mask)
            segment_ids.append(seg_ids)
            valid_positions.append(valid_pos)

        sequence_lengths = np.array([len(i) for i in input_ids])
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, padding='post')
        input_mask = tf.keras.preprocessing.sequence.pad_sequences(input_mask, padding='post')
        segment_ids = tf.keras.preprocessing.sequence.pad_sequences(segment_ids, padding='post')
        valid_positions = tf.keras.preprocessing.sequence.pad_sequences(valid_positions, padding='post')
        return input_ids, input_mask, segment_ids, valid_positions, sequence_lengths
