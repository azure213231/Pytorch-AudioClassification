# Pytorch声音分类模型



## 前言

项目主框架来自于[yeyupiaoling](https://github.com/yeyupiaoling)

https://github.com/yeyupiaoling/AudioClassification-Pytorch



### 基础使用

详情见https://github.com/yeyupiaoling/AudioClassification-Pytorch项目说明



### 新增



#### 预训练模型

增加pretrained_model文件夹，用于存放各个模型的预训练模型，可以通过引入相同框架中的预训练模型提高训练效果

```
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/res2net.yml',        '配置文件')
add_arg("local_rank",       int,    0,                          '多卡训练需要的参数')
add_arg("use_gpu",          bool,   True,                       '是否使用GPU训练')
add_arg('save_model_path',  str,    'models/',                  '模型保存的路径')
add_arg('resume_model',     str,    None,                       '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model', str,    'pretrained_model/res2net/res2net50_v1b_26w_4s-3cf99910.pth',                       '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args=args)
```



#### pth转pt、onnx

将训练好的pth模型转化为pt模型，可以用于进行模型迁移使用

```
model = Res2Net(input_size=_audio_featurizer.feature_dim,**configs.model_conf)
model.load_state_dict(model_state_dict)
model.eval()

# 转换为 TorchScript 格式
traced_model = torch.jit.trace(model, torch.randn(5, 512 ,80))
traced_model.save("output/pt/model.pt")
```

```
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
```
