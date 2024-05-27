from .hdd_data_layer import TRNHDDDataLayer
from .thumos_data_layer import TRNTHUMOSDataLayer
from .assembly101_data_layer import Assembly101Dataset

_DATA_LAYERS = {
    'TRNHDD': TRNHDDDataLayer,
    'TRNTHUMOS': TRNTHUMOSDataLayer,
    'Assembly': Assembly101Dataset
}

def build_dataset(args, phase):
    data_layer = _DATA_LAYERS[args.dataset]
    return data_layer(args, phase)
