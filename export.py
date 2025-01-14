import argparse
import torch
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.load_state import load_state


def convert_to_onnx(net, output_name):
    # input = torch.randn(1, 3, 320, 448)
    input = torch.randn(1, 3, 256, 288)
    # input = torch.randn(1, 3, 256, 256)
    input_names = ['data']
    output_names = ['stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

    torch.onnx.export(net, input, output_name, verbose=False,
                      input_names=input_names, output_names=output_names, opset_version=11)
    print('saved into: ', output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str,
                        required=True, help='path to the checkpoint')
    parser.add_argument('--output-name', type=str, default='human-pose-estimation.onnx',
                        help='name of output model in ONNX format')
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    convert_to_onnx(net, args.output_name)
