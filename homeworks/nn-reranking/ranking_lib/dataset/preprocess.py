import emoji
import nltk
import numpy as np
import re
import torch

from abc import ABC, abstractmethod

from transformers import AutoModel, AutoTokenizer, AutoConfig
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class Preprocessor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def process(self, data_element: dict):
        pass
                

class BasePreprocessor(Preprocessor):
    def __init__(
            self,
            model_name: str,
            model_type: str = "cross_encoder",
            concat_title_strategy: str = "space"
        ):
        super().__init__()
        nltk.download('stopwords')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.concat_title_strategy = concat_title_strategy
        self.filtered_stop_words = [
            word for word in stopwords.words('russian')
            if word not in {'как', 'что', 'кто', 'когда', 'где', 'почему', 'зачем',
                            'куда', 'откуда', 'сколько', 'какой', 'какая', 'чей', 'чья',
                            'чьё', 'чьи', 'чего'}
        ]+["null"]

        config = AutoConfig.from_pretrained(model_name)
        self.model_max_tokens = config.max_position_embeddings-2
        self.num_sep_tokens = (np.array(self.tokenizer.encode("abcd", "abcd")) == self.tokenizer.sep_token_id).sum()-1
        self.model_type = model_type

    @staticmethod
    def __remove_extra_spaces(text: str):
        text = re.sub(r'\s*([!?:;,.])\s*', r'\1 ', text) # Ппробелы перед знаками препинания
        text = re.sub(r"(?<=[(\[])\s+", "", text) # Пробелы после открывающих скобок
        text = re.sub(r"\s+(?=[)\]])", "", text)  # Пробелы перед закрывающими скобками и кавычками
        text = re.sub(r'["\'«“]\s*(.*?)\s*["\'»”]', r'"\1"', text)  # Удаление пробелов внутри кавычек
        return text.strip()


    def process_text(self, text: str):
        text = text.strip().split(" ")
        text = " ".join([w.lower() for w in text if not w.lower() in self.filtered_stop_words])
        text = emoji.replace_emoji(text, replace='')
        # text = self.__remove_extra_spaces(text)
        return text
    
    def concat_title(self, title: str | None, text: str):
        if title is None or self.concat_title_strategy == "none":
            if self.concat_title_strategy != "none":
                raise ValueError(f"Concat title strategy is {self.concat_title_strategy}, but title is None")
            return text
        if self.concat_title_strategy == "sep":
            return (self.tokenizer.sep_token*self.num_sep_tokens).join([title, text])
        elif self.concat_title_strategy == "space":
            return " ".join([title, text])
        else:
            raise ValueError("Unknown concat_title_strategy")

    def process(self, data_element: dict):
        if self.model_type == "cross_encoder":
            processed_element = {
                "qid": data_element["qid"],
                "label": data_element["label"],
                "item": self.tokenizer(
                    self.process_text(data_element["query"]),
                    self.concat_title(
                        self.process_text(data_element["title"]) if "title" in data_element else None,
                        self.process_text(data_element["text"])
                    ),
                    padding="max_length",
                    return_tensors="pt",
                    max_length=self.model_max_tokens,
                    truncation=True
                ),
            }
            processed_element["item"] = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v
                for k, v in processed_element["item"].items()
            }
        else:
            processed_element = {
                "qid": data_element["qid"],
                "label": data_element["label"],
                "query": self.tokenizer(
                    self.process_text(data_element["query"]),
                    padding="max_length",
                    return_tensors="pt",
                    max_length=self.model_max_tokens,
                    truncation=True
                ),
                "text": self.tokenizer(
                    self.concat_title(
                        self.process_text(data_element["title"]) if "title" in data_element else None,
                        self.process_text(data_element["text"])
                    ),
                    padding="max_length",
                    return_tensors="pt",
                    max_length=self.model_max_tokens,
                    truncation=True
                )
            }
            processed_element["query"] = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v
                for k, v in processed_element["query"].items()
            }
            processed_element["text"] = {
                k: v.squeeze() if isinstance(v, torch.Tensor) else v
                for k, v in processed_element["text"].items()
            }
        return processed_element


        return processed_element
