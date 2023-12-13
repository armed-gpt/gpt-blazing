from typing import Any

from .model import Baichuan2ModelConfig, Baichuan2Model, EmptyInitOnDevice


def convert_hf_model_to_model(hf_model: Any):
    with EmptyInitOnDevice():
        model = Baichuan2Model(Baichuan2ModelConfig(debug=True))
        model.half()

    baichuan_model = hf_model.model

    model.embed_tokens.load_state_dict(baichuan_model.embed_tokens.state_dict())
    for layer_idx, layer in enumerate(model.layers):
        layer.load_state_dict(baichuan_model.layers[layer_idx].state_dict())
    model.norm.load_state_dict(baichuan_model.norm.state_dict())
    model.lm_head.load_state_dict(hf_model.lm_head.state_dict())
    return model
