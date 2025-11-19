import torch.nn as nn
import sample_model
import json

def to_annat(model, f):
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
 
 if architechture_type == 'Sequential':
  layers = {}
  layers_count = 1
  layers_with_no_attribute = 0
  activations = 0

  for layer in architechture:
   try:
    attrs = {}

    for attribute in dir(layer):
     if str(type(getattr(layer, attribute))) == "<class 'int'>" and attribute != '_version':
      attrs.update({attribute:getattr(layer, attribute)})

    if not isinstance(layer, activations_tuple):
     if attrs:
      layers.update({f'layer{layers_count - layers_with_no_attribute - activations}':{'layer_defination':{"layer_name":type(layer).__name__, "layer_attrs":attrs}, 'activation':type(architechture[layers_count]).__name__ if isinstance(architechture[layers_count], activations_tuple) else None}})
     
     else:
      layers_with_no_attribute += 1

    else:
     activations += 1

   except IndexError:
    pass
  
   layers_count += 1

  json.dump({'ModelName':model_name, 'architechture':{'layers':layers}}, f, indent = 4)
 

model = sample_model.model

with open('model_json.json', 'w') as  f:
 to_annat(model, f)
