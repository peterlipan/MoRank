import torch.nn as nn
import torch.nn.functional as F
from .backbone import get_encoder
from .utils import ModelOutputs


class SoftmaxClassifier(nn.Module):
    def __init__(self, args):
        super(SoftmaxClassifier, self).__init__()

        self.encoder = get_encoder(args)
        self.d_hid = args.d_hid if hasattr(args, 'd_hid') else self.encoder.d_hid
        self.n_classes = args.n_classes
        self.head = nn.Linear(self.d_hid, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data):

        features = self.encoder(data['data'])
        logits = self.head(features)  # [B, T]
        probs = F.softmax(logits, dim=-1)
        y_pred = probs.argmax(dim=1)  # [B]
 

        return ModelOutputs(features=features,
                            logits=logits,
                            probs=probs,
                            y_pred=y_pred)

    def compute_loss(self, outputs, data):
        return self.criterion(outputs.logits, data['label'])
