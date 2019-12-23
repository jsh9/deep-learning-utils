import torch
import typeguard
import transformers # https://github.com/huggingface/transformers

from typing import List, Optional, Tuple

#%%----------------------------------------------------------------------------
def get_model_and_tokenizer(
        model_type: str, pretrained_name: Optional[str] = None
) -> Tuple[torch.nn.Module, transformers.PreTrainedTokenizer]:
    """
    Get pre-trained transfomer model and tokenizer.

    Parameters
    ----------
    model_type : {'BERT', 'DistilBERT', 'CTRL', 'TransfoXL', 'XLNet', 'XLM', 'RoBERTa', 'GPT2', 'OpenAIGPT'}
        The type of transformer model that you want.
    pretrained_name : Optional[str]
        Pretrained model name. See https://huggingface.co/transformers/pretrained_models.html
        for all valid names. If ``None``, the basic pretrained model
        correponding to each model type will be used.

    Returns
    -------
    model : torch.nn.Module
        The pre-trained transformer model.
    tokenizer :
        The pre-trained transformer tokenizer.

    Reference
    ---------
    https://github.com/huggingface/transformers#online-demo
    """
    typeguard.check_argument_types()

    model_pack = get_model_pack(model_type, pretrained_name)
    model = model_pack['model'].from_pretrained(model_pack['weights'])
    tokenizer = model_pack['tokenizer'].from_pretrained(model_pack['weights'])

    typeguard.check_return_type((model, tokenizer))
    return model, tokenizer

#%%----------------------------------------------------------------------------
def get_model_pack(model_type: str, pretrained_name: Optional[str] = None) -> dict:
    """
    Get the desired transformer "model pack".

    Parameters
    ----------
    model_type : {'BERT', 'DistilBERT', 'CTRL', 'TransfoXL', 'XLNet', 'XLM', 'RoBERTa', 'GPT2', 'OpenAIGPT'}
        The type of transformer model that you want.
    pretrained_name : Optional[str]
        Pretrained model name. See https://huggingface.co/transformers/pretrained_models.html
        for all valid names. If ``None``, the basic pretrained model
        correponding to each model type will be used.

    Returns
    -------
    model_pack : dict
        A dictionary of three keys: "model", "tokenizer", and "weights". You
        need to call `.from_pretrained()` on the values of "model" and
        "tokenizer" to get the actual model and the actual tokenizer.
    """
    typeguard.check_argument_types()

    if model_type == "BERT":
        model_pack =  _build_dict(
            transformers.BertModel,
            transformers.BertTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "DistilBERT":
        model_pack = _build_dict(
            transformers.DistilBertModel,
            transformers.DistilBertTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "CTRL":
        model_pack = _build_dict(
            transformers.CTRLModel,
            transformers.CTRLTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "TransfoXL":
        model_pack = _build_dict(
            transformers.TransfoXLModel,
            transformers.TransfoXLTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "XLNet":
        model_pack = _build_dict(
            transformers.XLNetModel,
            transformers.XLNetTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "XLM":
        model_pack = _build_dict(
            transformers.XLMModel,
            transformers.XLMTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "RoBERTa":
        model_pack = _build_dict(
            transformers.RobertaModel,
            transformers.RobertaTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "OpenAIGPT":
        model_pack = _build_dict(
            transformers.OpenAIGPTModel,
            transformers.OpenAIGPTTokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    elif model_type == "GPT2":
        model_pack = _build_dict(
            transformers.GPT2Model,
            transformers.GPT2Tokenizer,
            _get_default_pretrained_name(model_type, pretrained_name)
        )
    else:
        raise ValueError("Invalid `model_type`.")
    # END IF
    return model_pack

#%%----------------------------------------------------------------------------
def _get_default_pretrained_name(
        model_type: str, pretrained_name: Optional[str]
) -> str:
    default_pretrained_name = {
        'BERT': 'bert-base-uncased',
        'DistilBERT': 'distilbert-base-uncased',
        'CTRL': 'ctrl',
        'TransfoXL': 'transfo-xl-wt103',
        'XLNet': 'xlnet-base-cased',
        'XLM': 'xlm-mlm-enfr-1024',
        'RoBERTa': 'roberta-base',
        'GPT2': 'gpt2',
        'OpenAIGPT': 'openai-gpt'
    }
    if pretrained_name is None:
        return default_pretrained_name[model_type]
    # END IF
    return pretrained_name

#%%----------------------------------------------------------------------------
def _build_dict(transf_model, model_tokenizer, weights):
    result = {"model": transf_model,
              "tokenizer": model_tokenizer,
              "weights": weights}
    return result

#%%----------------------------------------------------------------------------
def tokenize_texts(
        texts: List[str],
        tokenizer: transformers.PreTrainedTokenizer
) -> List[List[int]]:
    """
    Tokenize a list of texts into their respective token IDs.

    Parameters
    ----------
    texts : List[str]
        List of texts to be tokenized.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use.

    Returns
    -------
    tokenized : List[List[int]]
        The list of token IDs corresponding to each line of text.
    """
    typeguard.check_argument_types()

    tokenized = []
    for text in texts:
        tokenized.append(tokenizer.encode(text, add_special_tokens=True))
    # END FOR

    typeguard.check_return_type(tokenized)
    return tokenized

#%%----------------------------------------------------------------------------
def tokenize_sentence_pairs(
        list_of_sentence_pairs: List[Tuple[str, str]],
        tokenizer: transformers.PreTrainedTokenizer
) -> List[List[int]]:
    """
    Tokenize a list of text pairs into a their respective token IDs.

    Parameters
    ----------
    list_of_sentence_pairs : List[Tuple[str, str]]
        List of sentence pairs. For example, [('hello world', 'good morning')]
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer to use.

    Returns
    -------
    tokenized :List[List[int]]
        The list of token IDs corresponding to each sentence pair.
    """
    typeguard.check_argument_types()

    tokenized = []
    for sentence_1, sentence_2 in list_of_sentence_pairs:
        this_tokenized = tokenizer.encode(
            sentence_1, text_pair=sentence_2, add_special_tokens=True
        )
        tokenized.append(this_tokenized)
    # END FOR

    typeguard.check_return_type(tokenized)
    return tokenized
