from __future__ import print_function

import sys

sys.path.append("../config")
sys.path.append("../blazeface")

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg_mnet, cfg_slim, cfg_rfb, cfg_blaze
from models.module.prior_box import PriorBox
from models.module.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from models.net_blaze import Blaze
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import onnx

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model',
                    default='../weights/pretrain/Blaze_Final_640.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='Blaze',
                    help='Backbone network mobile0.25 or slim or RFB')
# parser.add_argument('--model_size', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--size', default=(640, 368), help='hw')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')

args = parser.parse_args()


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path,
                                     map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path,
                                     map_location=lambda storage, loc: storage.cuda(
                                         device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg=cfg, phase='test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg=cfg, phase='test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg=cfg, phase='test')
    elif args.network == "Blaze":
        cfg = cfg_blaze
        net = Blaze(cfg=cfg, phase='test')
    else:
        print("Don't support network!")
        exit(0)

    # load weight
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    ##################export###############
    f = f'../weights/blaceface_{m_h}x{m_w}.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(f))
    input_names = ["input"]
    output_names = ["boxes", 'scores', 'landmark']
    inputs = torch.randn(1, 3, m_h, m_w).to(device)
    torch_out = torch.onnx.export(net, inputs, f, export_params=True,
                                  verbose=False,
                                  opset_version=11,
                                  input_names=input_names, output_names=output_names)

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    try:
        import onnxsim

        print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
        onnx_model, check = onnxsim.simplify(onnx_model,
                                             dynamic_input_shape=False,
                                             input_shapes={'input': list(
                                                 inputs.shape)})
        assert check, "simplify check failed "
        onnx.save(onnx_model, f)
    except Exception as e:
        print(f"simplifer failure: {e}")

    import onnxruntime

    providers = ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(f, providers=providers)
    detect_inputs = {session.get_inputs()[0].name: inputs.numpy()}
    y_onnx = session.run(None,
                         detect_inputs)
    y_net = net(inputs)
    print("torch pred's shape is ", [_x.shape for _x in y_onnx])
    print("onnx pred's shape is ", [_x.shape for _x in y_onnx])
    for _idx, conten in enumerate(y_onnx):
        print(
            f"{_idx} ,cos_sim is {cos_sim(y_onnx[_idx].flatten(), y_net[_idx].flatten())}")
        ##################end###############
