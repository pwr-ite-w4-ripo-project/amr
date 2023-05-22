import torch
import cv2

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

def run(
        weights,
        source,
        confidence_thres,
        imgsz=(640, 640)
):
    model = DetectMultiBackend(weights, device=select_device('cpu'))
    stride, names, pt = model.stride, model.names, model.pt

    dataset = LoadImages(str(source), img_size=imgsz, stride=stride, auto=pt)

    for _, im, im0s, *_ in dataset:
        im = torch.from_numpy(im).to(model.device)
        im = im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
        pred = non_max_suppression(pred, confidence_thres)

        # Process predictions
        for _, det in enumerate(pred):
            im0 = im0s.copy()

            annotator = Annotator(im0, line_width=3, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            cv2.imshow('', im0)
            cv2.waitKey(1)


if __name__ == '__main__':
    run(weights='model.pt', source='10_czlowiek_karton.mp4', confidence_thres=0.25)
