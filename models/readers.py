# -*- coding: utf-8 -*-
import os


# Reader class for reading different type of dataset
class Reader:

    def __init__(self):
        pass

    @staticmethod
    def read_goo_format(dataset_folder_path: str):
        labels = None
        text_arr = None
        tags_arr = None
        with open(os.path.join(dataset_folder_path, 'label'), encoding='utf-8') as f:
            labels = f.readlines()

        with open(os.path.join(dataset_folder_path, 'seq.in'), encoding='utf-8') as f:
            text_arr = f.readlines()

        with open(os.path.join(dataset_folder_path, 'seq.out'), encoding='utf-8') as f:
            tags_arr = f.readlines()

        assert len(text_arr) == len(tags_arr) == len(labels)
        return text_arr, tags_arr, labels


if __name__ == '__main__':
    text_arr_, tags_arr_, labels_ = Reader.read_goo_format('models/dataset/snips/valid')
    print(text_arr_[:5])
    print(tags_arr_[:5])
    print(labels_[:5])