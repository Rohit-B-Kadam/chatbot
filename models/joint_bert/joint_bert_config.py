config = {
    "train": "models/dataset/snips/train",
    "val": "models/dataset/snips/valid",
    "test": "models/dataset/snips/test",
    "save": "models/saved_models/joint_bert_model",
    "epochs": 7,  # 5
    "batch": 64,
    "type": "bert",  # [bert | albert]
    "hub_url": {
        "bert": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        "albert": "https://tfhub.dev/tensorflow/albert_en_base/1"
    },
    "start_model": None  # [None | path save model]
}

# config info:
#   train: Path to training data in Goo et al format
#   val: Path to validation data in Goo et al format
#   test: Path to test data in Goo et al format
#   save: Folder path to save the trained model
#   epochs: Number of epochs
#   batch: Batch size
#   type : which model to train option [bert | albert]
#   hub_url : url of bert and albert model (don't change it)
#   start_model : Path to joint BERT / ALBERT NLU model for incremental training
##

# bert version2 https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2
#