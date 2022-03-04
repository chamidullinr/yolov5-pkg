import torch
import torch.nn as nn
from torchvision import transforms

from yolov5.utils.general import non_max_suppression



class ModelWithPreProcessing(nn.Module):
    def __init__(self, base_model, nms_kwargs={}):
        super().__init__()
        self.base_model = base_model
        self.nms_kwargs = nms_kwargs

    def _nms(self, pred):
        pred = non_max_suppression(pred, **self.nms_kwargs)
        pred = torch.cat(pred, dim=0)
        return pred

    def forward(self, img):
        # image = image.to(torch.float)
        # image = image.permute(2, 0, 1)
        # batch = image.unsqueeze(0)
        # batch = batch / 255.
        # batch = torch.nn.functional.interpolate(batch, size=self._input_size, mode='bilinear', align_corners=True)
        # batch = self._transform_normalize(batch)
        # x = self._base_model(batch.to(self._device))
        # TODO - add pre-processing

        x = self.base_model(img)
        # x = (self._nms(x[0]), )  # TODO - fix

        return x
