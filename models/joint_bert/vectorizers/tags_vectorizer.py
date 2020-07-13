# -*- coding: utf-8 -*-
from sklearn.preprocessing import LabelEncoder
import numpy as np


class TagsVectorizer:

    def __init__(self):
        self.label_encoder = LabelEncoder()

    @staticmethod
    def tokenize(tags_str_arr):
        return [s.split() for s in tags_str_arr]

    def fit(self, tags_str_arr):
        data = ['<PAD>'] + [item for sublist in self.tokenize(tags_str_arr) for item in sublist]
        self.label_encoder.fit(data)

    def transform(self, tags_str_arr, valid_positions):
        seq_length = valid_positions.shape[1]  # .shape[0]: number of rows, .shape[1]: number of columns
        data = self.tokenize(tags_str_arr)
        # [Ignore] we added the 'CLS' and 'SEP' token as the first and last token for every sentence respectively
        data = [self.label_encoder.transform(['O'] + x + ['O']).astype(np.int32) for x in data]

        output = np.zeros((len(data), seq_length))
        for i in range(len(data)):
            idx = 0
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    output[i][j] = data[i][idx]
                    idx += 1
        return output

    def inverse_transform(self, model_output_3d, valid_positions):
        seq_length = valid_positions.shape[1]
        slots = np.argmax(model_output_3d, axis=-1)
        slots = [self.label_encoder.inverse_transform(y) for y in slots]
        output = []
        for i in range(len(slots)):
            y = []
            for j in range(seq_length):
                if valid_positions[i][j] == 1:
                    y.append(str(slots[i][j]))
            output.append(y)
        return output

    def load(self):
        pass

    def save(self):
        pass


if __name__ == '__main__':
    tags_str_arr_ = ['O O B-X B-Y', 'O B-Y O']
    valid_positions_ = np.array([[1, 1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1, 0, 1]])

    vectorizer = TagsVectorizer()
    vectorizer.fit(tags_str_arr_)
    data_ = vectorizer.transform(tags_str_arr_, valid_positions_)
    print(data_)
    print(vectorizer.label_encoder.classes_)
