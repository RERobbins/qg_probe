import os
import numpy as np
import pandas as pd

import torch

from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# ### Probe Class
#
# A probe instance can be used to generate predictions from any of our models.
#
# It loads (and caches) appropriate tokenizers and models as needed.
#
# The only method you need to use is the predict method.


class Probe:
    """
    A class used to represent an Awesome Puppies Question Generation Probe
    ...

    Attributes
    ----------
    model_root : str
        the pathname for the root of the model collection
    models : dict
        a dictionary of cached models
    tokenizers : dict
        a dictionary of cached tokenizers

    Methods
    -------
    predict(context, answer, base_model='T5', training_dataset='amalgam',
            num_beams=4, early_stopping=True, no_repeat_ngram_size=3,
            maximum_input_length=1024, maximum_target_length=50)

        returns a prediction string

    retrieve_tokenizer (base_model='T5')

        returns and caches a tokenizer

    retrieve_model(base_model='bart', training_dataset='amalgam')

        returns and caches a model
    """

    def __init__(self, model_root):
        self.model_root = model_root
        self.models = {}
        self.tokenizers = {}

    def predict(
        self,
        context,
        answer,
        base_model="T5",
        training_dataset="amalgam",
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        maximum_input_length=1024,
        maximum_target_length=50,
    ):

        tokenizer = self.retrieve_tokenizer(base_model)
        model = self.retrieve_model(base_model, training_dataset)

        if base_model == "bart":
            prompt_string = f"{answer} </s> {context}"
        elif base_model == "T5":
            prompt_string = f"generate question: answer: {answer} context: {context}"
        else:
            raise ValueError("invalid base model")

        inputs = tokenizer(
            prompt_string,
            return_tensors="pt",
            max_length=maximum_input_length,
            truncation=True,
            padding=True,
        )
        output_ids = model.generate(
            inputs["input_ids"].cuda(),
            max_length=maximum_target_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )
        prediction = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return prediction

    def retrieve_tokenizer(self, base_model="T5"):
        tokenizer = self.tokenizers.get(base_model)
        if tokenizer:
            return tokenizer

        if base_model == "bart":
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        elif base_model == "T5":
            tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
        else:
            raise ValueError("invalid base model")

        self.tokenizers[base_model] = tokenizer
        return tokenizer

    def retrieve_model(self, base_model="T5", training_dataset="amalgam"):
        model_tuple = (base_model, training_dataset)
        model = self.models.get(model_tuple)
        if model:
            return model

        model_dir = f"{self.model_root}{base_model}_base_pt_long.{training_dataset}"

        if base_model == "bart":
            print(f"Loading: {model_dir}")
            model = BartForConditionalGeneration.from_pretrained(model_dir)
        elif base_model == "T5":
            print(f"Loading: {model_dir}")
            model = T5ForConditionalGeneration.from_pretrained(model_dir)
        else:
            raise ValueError("invalid base model")

        model.to(torch.device("cuda:0"))
        self.models[model_tuple] = model
        return model


def panel(
    probe,
    context,
    answers,
    base_models,
    training_datasets,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=3,
    maximum_input_length=1024,
    maximum_target_length=50,
):
    result = [
        {
            "Base Model": base_model,
            "Training Dataset": training_dataset,
            "Answer": answer,
            "Prediction": "".join(
                probe.predict(
                    context,
                    answer,
                    base_model=base_model,
                    training_dataset=training_dataset,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                    maximum_input_length=maximum_input_length,
                    maximum_target_length=maximum_target_length,
                )
            ),
        }
        for answer in answers
        for base_model in base_models
        for training_dataset in training_datasets
    ]
    replacements = {
        "bart": "BART",
        "amalgam": "Shuffled Blended",
        "nq": "NQ",
        "quac": "QuAC",
        "squad": "SQuAD",
        "triviaqa": "TriviaQA",
    }
    return pd.DataFrame(result).replace(replacements)
