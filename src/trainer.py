from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch
import numpy as np

class VisionTrainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.criteria = nn.CrossEntropyLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        output = model(inputs['inputs'])
        label = inputs['labels']

        loss = self.criteria(output, label)

        if return_outputs:
            return loss, output, label
        
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(model,inputs,True)
        
        return (eval_loss,pred,label)