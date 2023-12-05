import torch
import torch.utils._device

from .model import Baichuan2Model, Baichuan2ModelConfig


# http://www.lernapparat.de/faster-model-init
class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):  # type: ignore

    def __init__(self, device=None):  # type: ignore
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore
        kwargs = kwargs or {}
        if getattr(func, '__module__', None) == 'torch.nn.init':
            if 'tensor' in kwargs:
                return kwargs['tensor']
            else:
                return args[0]
        device_constructors = torch.utils._device._device_constructors()  # type: ignore
        if (
            self.device is not None and func in device_constructors and kwargs.get('device') is None
        ):
            kwargs['device'] = self.device
        return func(*args, **kwargs)


def debug_compile():
    with EmptyInitOnDevice():
        model = Baichuan2Model(Baichuan2ModelConfig())
    model.bfloat16()

    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.to_devices([('cuda:0', 0), ('cuda:1', 20)])  # type: ignore
    breakpoint()
