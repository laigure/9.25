"""Reproduce data/class_embeddings3d.npy with this checkout's roberta_base.

Same as gen_class_embeddings.py, except the hard-coded author paths are
replaced by the local data/roberta_base directory (run from the 3EED root).
"""

import numpy as np
from transformers import RobertaModel, RobertaTokenizerFast

from data.model_util_scannet import ScannetDatasetConfig


def main():
    config = ScannetDatasetConfig()
    tokenizer = RobertaTokenizerFast.from_pretrained("data/roberta_base")
    text_encoder = RobertaModel.from_pretrained("data/roberta_base")

    object_list = [config.class2type[i] for i in range(len(config.class2type))]
    tokenized = tokenizer.batch_encode_plus(object_list, padding="longest", return_tensors="pt")
    encoded_text = text_encoder(**tokenized)
    object_embeddings = (
        encoded_text.last_hidden_state
        * tokenized.attention_mask.unsqueeze(-1)
        / tokenized.attention_mask.sum(-1)[:, None, None]
    ).sum(1)
    np.save("data/class_embeddings3d.npy", object_embeddings.detach().numpy())
    print("saved", object_embeddings.shape)


if __name__ == "__main__":
    main()
