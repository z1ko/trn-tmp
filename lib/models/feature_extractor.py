import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class HDDFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(HDDFeatureExtractor, self).__init__()

        if args.inputs in ['camera', 'sensor', 'multimodal']:
            self.with_camera = 'sensor' not in args.inputs
            self.with_sensor = 'camera' not in args.inputs
        else:
            raise(RuntimeError('Unknown inputs of {}'.format(args.inputs)))

        if self.with_camera and self.with_sensor:
            self.fusion_size = 1280 + 20
        elif self.with_camera:
            self.fusion_size = 1280
        elif self.with_sensor:
            self.fusion_size = 20

        self.camera_linear = nn.Sequential(
            nn.Conv2d(1536, 20, kernel_size=1),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        self.sensor_linear = nn.Sequential(
            nn.Linear(8, 20),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_input, sensor_input):
        if self.with_camera:
            camera_input = self.camera_linear(camera_input)
        if self.with_sensor:
            sensor_input = self.sensor_linear(sensor_input)

        if self.with_camera and self.with_sensor:
            fusion_input = torch.cat((camera_input, sensor_input), 1)
        elif self.with_camera:
            fusion_input = camera_input
        elif self.with_sensor:
            fusion_input = sensor_input
        return fusion_input

class THUMOSFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(THUMOSFeatureExtractor, self).__init__()

        if args.inputs in ['camera', 'motion', 'multistream']:
            self.with_camera = 'motion' not in args.inputs
            self.with_motion = 'camera' not in args.inputs
        else:
            raise(RuntimeError('Unknown inputs of {}'.format(args.inputs)))

        if self.with_camera and self.with_motion:
            self.fusion_size = 2048 + 1024
        elif self.with_camera:
            self.fusion_size = 2048
        elif self.with_motion:
            self.fusion_size = 1024

        self.input_linear = nn.Sequential(
            nn.Linear(self.fusion_size, self.fusion_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, camera_input, motion_input):
        if self.with_camera and self.with_motion:
            fusion_input = torch.cat((camera_input, motion_input), 1)
        elif self.with_camera:
            fusion_input = camera_input
        elif self.with_motion:
            fusion_input = motion_input
        return self.input_linear(fusion_input)

class AssemblyFeatureExtractor(nn.Module):
    def __init__(self, args):
        super(AssemblyFeatureExtractor, self).__init__()
        self.modality = args.modality

        if self.modality == 'embeddings':
            self.fusion_size = 2048
        else:
            self.fusion_size = 42*3

        self.input_linear = nn.Sequential(
            nn.Linear(self.fusion_size, self.fusion_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, data):
        if self.modality == 'embeddings':
            x = data
        else:
            x = data

        return self.input_linear(x)

_FEATURE_EXTRACTORS = {
    'HDD': HDDFeatureExtractor,
    'THUMOS': THUMOSFeatureExtractor,
    'Assembly': AssemblyFeatureExtractor
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)
