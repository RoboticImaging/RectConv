# --------------------------------------------------
# Author: Ryan Griffiths (r.griffiths@sydney.edu.au)
# Run RectConv on woodscape with different networks
# --------------------------------------------------

import argparse
import torch
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50
from torchvision import transforms

import data_loaders.woodscape_loader as woodscape_loader
from scripts.generate_metrics import generate_metrics 
from scripts.util import generate_offset, convert_to_rectconv
from scripts.projection import read_cam_from_json
import scripts.network as deeplabv3plus


def get_args():
    """ Get args required to run tests.
    """

    parser = argparse.ArgumentParser('RectConv', add_help=False)
    parser.add_argument('--model', default='deeplabv3plus_resnet101', type=str,
                        help='Model to evaluate [fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3plus_resnet101] (default: %(default)s)')
    parser.add_argument('--camera', default='MVR', type=str,
                        help='Which camera from woodscape to use [MVR, MVL](default: %(default)s)')
    parser.add_argument('--data_path', type=str, help='Path to woodscape dataset')
    parser.add_argument('--model_checkpoints', type=str, help='Path to model checkpoints')
    return parser.parse_args()


def main(args):
    """ Run tests on the woodscape dataset, comparing convolutions with RectConvs
    """

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    _, test_dl = woodscape_loader.load(args.data_path, 1, args.camera, transform)

    fisheye_cam = read_cam_from_json(args.data_path+'/calibration/'+args.camera+'.json')
    offset = generate_offset(fisheye_cam)

    if args.model == 'deeplabv3plus_resnet101':
        model = deeplabv3plus.modeling.__dict__[args.model](num_classes=19, output_stride=16, pretrained_backbone=False)
    else:
        model = eval(args.model+'(num_classes=19, weights_backbone=None)')
    model.load_state_dict( torch.load(args.model_checkpoints))
    
    print('|------ {} - {} ------|'.format(args.model, args.camera))
    metricsCNN = generate_metrics(model, test_dl, mask=args.camera)
    print('|------ Orig Conv ------|')
    print("Pixel Acc: {:.2f},   MIOU: {:.2f}".format(metricsCNN['acc']*100, metricsCNN['miou']*100))

    convert_to_rectconv(model, offset)
    metricsRectifyCNN = generate_metrics(model, test_dl, mask=args.camera)
    print('|------ RectConv  ------|')
    print("Pixel Acc: {:.2f},   MIOU: {:.2f}".format(metricsRectifyCNN['acc']*100, metricsRectifyCNN['miou']*100))


if __name__ == '__main__':
    args = get_args()
    main(args)