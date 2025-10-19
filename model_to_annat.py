import torch.nn as nn
import json

def to_json(model, f):
 activations_tuple = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.PReLU,
    nn.RReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Softmax,     # dim must be specified
    nn.Softmax2d,
    nn.Softplus,
    nn.Softshrink,
    nn.Hardtanh,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.CELU,
    nn.GELU,
    nn.SELU,
    nn.SiLU,
    nn.Mish
 )

 model_architechture_define = model.modules()

 model_name = next(model_architechture_define)._get_name()
 architechture_type = next(model_architechture_define)._get_name()
 architechture = list(model_architechture_define)

 layers = {}
 layers_count = 1
 activations = 0

 for layer in architechture:
  try:
   if not isinstance(layer, activations_tuple):
    layers.update({f'layer{layers_count - activations}':{'layer_defination':str(layer), 'activation':str(architechture[layers_count]) if isinstance(architechture[layers_count], activations_tuple) else None}})
  
   else:
    activations += 1

   layers_count += 1

  except IndexError:
   pass

  json.dump({'ModelName':model_name, 'architechture':{'architechture_type':architechture_type, 'layers':layers}}, f, indent = 4)