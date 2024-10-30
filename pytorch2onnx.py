import torch.onnx
from macls.data_utils.featurizer import AudioFeaturizer
from macls.models.campplus import CAMPPlus
from macls.models.ecapa_tdnn import EcapaTdnn
from macls.models.res2net import Res2Net
from macls.models.resnet_se import ResNetSE

# _audio_featurizer = AudioFeaturizer(feature_method="Fbank",
#                                         method_args={'sample_frequency': 16000, 'num_mel_bins': 80})

base_path = 'model/EcapaTdnn-16k-训练60次-288条/'
# 转为ONNX
def Convert_ONNX(model):
    # 设置模型为推理模式
    model.eval()

    # 设置模型输入的尺寸
    dummy_input = torch.randn(5 , 512 , 80, requires_grad=True)

    dynamic_axes = {'modelInput': {0: 'batch', 1 : 'height'}, 'modelOutput': {0: 'batch'}}

    # 导出ONNX模型
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      base_path + "onnx/best.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the onnx version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes=dynamic_axes)
    print(" ")
    print('Model has been converted to onnx')


if __name__ == "__main__":
    # 构建模型并训练
    # xxxxxxxxxxxx

    # 测试模型精度
    # testAccuracy()

    # 加载模型结构与权重
    # model = CAMPPlus(input_size=_audio_featurizer.feature_dim, **{'num_class': 2})
    path = base_path + "best_model/model.pth"
    # model_state_dict = torch.load(path)
    # model.load_state_dict(model_state_dict)

    # model = CAMPPlus(input_size=80, **{'num_class': 2})
    # model = Res2Net(input_size=80, **{'num_class': 2})
    # model = ResNetSE(input_size=80, **{'num_class': 2})
    model = EcapaTdnn(input_size=80, **{'num_class': 2})
    model_state_dict = torch.load(path)
    model.load_state_dict(model_state_dict)

    # 转换为ONNX
    Convert_ONNX(model)

