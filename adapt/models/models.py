import torch

models = {}
# Lenet, ResNet34, AdaptNet
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    print("models:")
    print(models)
    if torch.cuda.is_available():
        net = net.cuda()
    print("net")
    print(net)
    return net
