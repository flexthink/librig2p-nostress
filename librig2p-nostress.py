# coding=utf-8
# Copyright 2021 Artem Ploujnikov


# Lint as: python3
import json

import datasets

_DESCRIPTION = """\
Grapheme-to-Phoneme training, validation and test sets
"""

_TRAIN_URL = "https://raw.githubusercontent.com/flexthink/librig2p-nostress/develop/dataset/train.json"
_VALID_URL = "https://raw.githubusercontent.com/flexthink/librig2p-nostress/develop/dataset/valid.json"
_TEST_URL = "https://raw.githubusercontent.com/flexthink/librig2p-nostress/develop/dataset/train.json"

_HOMEPAGE_URL = "https://raw.githubusercontent.com/flexthink/speechbrain-g2pdataset"

_PHONEMES = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
_ORIGINS = ["librispeech", "timit"]


class GraphemeToPhoneme(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "origin": datasets.ClassLabel(names=_ORIGINS),
                    "char": datasets.Value("string"),
                    "phn": datasets.Sequence(datasets.ClassLabel(names=_PHONEMES)),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(_TRAIN_URL)
        valid_path = dl_manager.download_and_extract(_VALID_URL)
        test_path = dl_manager.download_and_extract(_TEST_URL)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": train_path, "datatype": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"datapath": valid_path, "datatype": "valid"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"datapath": test_path, "datatype": "test"},
            ),
        ]

    def _generate_examples(self, datapath, datatype):
        with open(datapath, encoding="utf-8") as f:
            data = json.load(f)

        for sentence_counter, (item_id, item) in enumerate(data.items()):
            resp = {
                "id": item["id"],
                "origin": item["origin"],
                "char": item["char"],
                "phn": item["phn"],
            }
            yield sentence_counter, resp
