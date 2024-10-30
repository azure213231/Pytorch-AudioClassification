import torch
import yaml

from macls.data_utils.featurizer import AudioFeaturizer
from macls.models.res2net import Res2Net
from macls.predict import MAClsPredictor
import argparse
import functools
from macls.models.ecapa_tdnn import EcapaTdnn
import os

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

def dict_to_object(dict_obj):
    if not isinstance(dict_obj, dict):
        return dict_obj
    inst = Dict()
    for k, v in dict_obj.items():
        inst[k] = dict_to_object(v)
    return inst

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('configs',          str,    'configs/res2net.yml',   '配置文件')
    add_arg("use_gpu",          bool,  True,                        "是否使用GPU评估模型")
    add_arg('model_path',       str,    'models/Res2Net_Fbank/best_model/', '导出的预测模型文件路径')
    args = parser.parse_args()
    print_arguments(args=args)

    # 定义并加载 PyTorch 模型
    resume_model = 'models/Res2Net_Fbank/best_model/model.pth';

    # 加载预训练模型
    if os.path.isdir(resume_model):
        resume_model = os.path.join(resume_model, 'model.pth')
    assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
    model_state_dict = torch.load(resume_model)

    device = torch.device("cuda")
    # 读取配置文件
    configs = args.configs
    if isinstance(configs, str):
        with open(configs, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        print_arguments(configs=configs)
    configs = dict_to_object(configs)

    # 获取特征器
    _audio_featurizer = AudioFeaturizer(feature_method=configs.preprocess_conf.feature_method,
                                             method_args=configs.preprocess_conf.get('method_args', {}))
    _audio_featurizer.to(device)
    # 获取分类标签
    with open(configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    class_labels = [l.replace('\n', '') for l in lines]
    # 自动获取列表数量
    if configs.model_conf.num_class is None:
        configs.model_conf.num_class = len(class_labels)

    # model = EcapaTdnn(input_size=_audio_featurizer.feature_dim,**configs.model_conf)
    # model.load_state_dict(model_state_dict)
    # model.eval()
    #
    model = Res2Net(input_size=_audio_featurizer.feature_dim,**configs.model_conf)
    model.load_state_dict(model_state_dict)
    model.eval()

    # 转换为 TorchScript 格式
    traced_model = torch.jit.trace(model, torch.randn(5, 512 ,80))
    traced_model.save("output/pt/model.pt")