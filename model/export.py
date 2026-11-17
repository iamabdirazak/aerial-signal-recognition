"""Export: the released classifier as an ONNX file, with a model card beside it.

The file holds the classification stage only. Sequencing, normalisation and gating stay in
Python, since they are light and depend on the stream rather than on learned weights.

    python -m model.export          # models/weights.pt -> models/model.onnx and models/model.json
"""
import json
import os
import platform
import time

import numpy as np
import torch

from .classification import SignalLSTM
from .constants import CLASSES, FEATURE_DIM, FPS, WINDOW
from .recogniser import DEFAULT_WEIGHTS

MODELS = os.path.dirname(DEFAULT_WEIGHTS)
INPUT, OUTPUT = 'windows', 'logits'


def to_onnx(weights=DEFAULT_WEIGHTS, path=os.path.join(MODELS, 'model.onnx')):
    net = SignalLSTM(FEATURE_DIM)
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.eval()
    torch.onnx.export(net, torch.zeros(1, WINDOW, FEATURE_DIM), path, input_names=[INPUT],
                      output_names=[OUTPUT], dynamic_axes={INPUT: {0: 'batch'}, OUTPUT: {0: 'batch'}},
                      opset_version=17, dynamo=False)
    return net, path


def verify(net, path, windows=256, repeats=500):
    """Agreement with PyTorch on random windows, and the time per window on one CPU thread."""
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(path, options, providers=['CPUExecutionProvider'])
    x = torch.randn(windows, WINDOW, FEATURE_DIM)
    with torch.no_grad():
        reference = net(x).numpy()
    logits = session.run(None, {INPUT: x.numpy()})[0]
    one = x[:1].numpy()
    for _ in range(50):
        session.run(None, {INPUT: one})
    start = time.perf_counter()
    for _ in range(repeats):
        session.run(None, {INPUT: one})
    return dict(largest_difference=float(np.abs(reference - logits).max()),
                same_decision=float((reference.argmax(1) == logits.argmax(1)).mean()),
                milliseconds_per_window=round((time.perf_counter() - start) / repeats * 1000, 3),
                machine=f'{platform.system()} {platform.machine()}')


def card(net, path, check):
    return {
        'name': 'Aerial Signal Recognition',
        'task': 'Aircraft-handling body signal from a window of 2D body keypoints',
        'input': {'name': INPUT, 'shape': ['batch', WINDOW, FEATURE_DIM], 'frames': WINDOW, 'rate_hz': FPS,
                  'features': '14 joints, body-centred position and velocity'},
        'output': {'name': OUTPUT, 'classes': CLASSES},
        'parameters': sum(p.numel() for p in net.parameters()),
        'file': {'name': os.path.basename(path), 'kilobytes': round(os.path.getsize(path) / 1024, 1)},
        'check': check,
        'training': 'UAV-GESTURE, all 119 videos, two-layer LSTM with viewpoint augmentation',
    }


def main():
    net, path = to_onnx()
    model_card = card(net, path, verify(net, path))
    with open(os.path.join(MODELS, 'model.json'), 'w') as f:
        json.dump(model_card, f, indent=2)
    print(json.dumps(model_card, indent=2))


if __name__ == '__main__':
    main()
