import torch.nn as nn

from transformers import AutoModel, AutoTokenizer

class CrossEncoder(nn.Module):
    def __init__(
            self,
            model_name: str,
            train_layers_count: int = 2,
            num_outputs: int = 1
        ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config

        # freeze all layers without bias and LN
        for name, par in self.bert.named_parameters():
            if 'bias' in name or 'LayerNorm' in name:
                continue
            par.requires_grad = False

        layer_count = self.config.num_hidden_layers
        for i in range(train_layers_count): #unfreeze somw layers
            for par in self.bert.encoder.layer[layer_count - 1 - i].parameters():
                par.requires_grad = True
        self.head = nn.Linear(self.config.hidden_size, 1)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.bert(input_ids=input_ids,
                      token_type_ids=token_type_ids,
                      attention_mask=attention_mask
                      )[0][:, 0, :] #hidden_state of [CLS]
        x = self.head(x)
        return x