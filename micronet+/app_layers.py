import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import self_ops

# ********************* observers *********************
class ObserverBase(nn.Module):
    def __init__(self, q_level):
        super(ObserverBase, self).__init__()
        self.q_level = q_level

    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        self.update_range(torch.min(input), torch.max(input))


class MinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels):
        super(MinMaxObserver, self).__init__(q_level)
        self.num_flag = 0
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = torch.min(min_val_cur, self.min_val)
            self.max_val = torch.max(max_val_cur, self.max_val)


class MovingAverageMinMaxObserver(ObserverBase):
    def __init__(self, q_level, out_channels, momentum=0.1):
        super(MovingAverageMinMaxObserver, self).__init__(q_level)
        self.momentum = momentum
        self.num_flag = 0
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

    def update_range(self, min_val_cur, max_val_cur):
        if self.num_flag == 0:
            self.num_flag += 1
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur


# ********************* quantizers *********************
class AsymmetricQuantizer(nn.Module):
    def __init__(self, bits, observer, activation_weight_flag=False, qaft=False, union=False):
        super(AsymmetricQuantizer, self).__init__()
        self.observer = observer
        self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("eps", torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32))
        self.register_buffer("quant_max_val", torch.tensor(((1 << bits) - 1), dtype=torch.float32))
        self.register_buffer("zero_point_neg", torch.zeros((1), dtype=torch.float32))

    def forward(self, input):
        if self.training:
            self.observer(input) 
            self.scale = torch.max((self.observer.max_val - self.observer.min_val) / self.quant_max_val, self.eps) 
            self.zero_point_neg = torch.round(self.observer.min_val / self.scale)
        quant_val = input / self.scale - self.zero_point_neg
        quant_val = quant_val - quant_val.detach() + torch.round(quant_val)
        output = ( torch.clamp(quant_val, 0, self.quant_max_val) + self.zero_point_neg ) * self.scale
        return output


# ********************* layers *********************
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, a_bits=8, w_bits=8):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.activation_observer = MovingAverageMinMaxObserver(q_level="L", out_channels=None)
        self.weight_observer = MinMaxObserver(q_level="L", out_channels=None)
        self.eps = torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)

        self.activation_quant_max = torch.tensor(((1 << a_bits) - 1), dtype=torch.float32)
        self.register_buffer("activation_scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("activation_zp_neg", torch.zeros((1), dtype=torch.float32))

        self.weight_quant_max = torch.tensor(((1 << w_bits) - 1), dtype=torch.float32)
        self.register_buffer("weight_scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("weight_zp_neg", torch.zeros((1), dtype=torch.float32))

    def quantize(self, x, s, zpn, qmax):
        x_affine = x / s - zpn
        x_int = torch.clamp(x_affine - x_affine.detach() + torch.round(x_affine), 0, qmax)
        return (x_int + zpn) * s

    def forward(self, input):
        # update activation and weight range
        if self.training:
            self.activation_observer(input)
            self.activation_scale = torch.max((self.activation_observer.max_val - self.activation_observer.min_val) / self.activation_quant_max, self.eps)
            self.activation_zp_neg = torch.round(self.activation_observer.min_val / self.activation_scale)
            self.weight_observer(self.weight)
            self.weight_scale = torch.max((self.weight_observer.max_val - self.weight_observer.min_val) / self.weight_quant_max, self.eps)
            self.weight_zp_neg = torch.round(self.weight_observer.min_val / self.weight_scale)
        
        # accurate computation
        input_fq = self.quantize(input, self.activation_scale, self.activation_zp_neg, self.activation_quant_max)
        weight_fq = self.quantize(self.weight, self.weight_scale, self.weight_zp_neg, self.weight_quant_max)
        output = F.linear(input_fq, weight_fq, self.bias)

        return output

        
class ApproxConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        a_bits=8,
        w_bits=8,
        use_ste_gradient=False
    ):
        super(ApproxConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.activation_observer = MovingAverageMinMaxObserver(q_level="L", out_channels=None)
        self.weight_observer = MinMaxObserver(q_level="L", out_channels=None)
        self.eps = torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)

        self.activation_quant_max = torch.tensor(((1 << a_bits) - 1), dtype=torch.float32)
        self.register_buffer("activation_scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("activation_zp_neg", torch.zeros((1), dtype=torch.float32))

        self.weight_quant_max = torch.tensor(((1 << w_bits) - 1), dtype=torch.float32)
        self.register_buffer("weight_scale", torch.ones((1), dtype=torch.float32))
        self.register_buffer("weight_zp_neg", torch.zeros((1), dtype=torch.float32))

        self.conv2d_op = self_ops.approx_conv2d_baseline_op if use_ste_gradient else self_ops.approx_conv2d_op

    def quantize_v2(self, x, s, zpn, qmax):
        x_affine = x / s - zpn
        x_int = torch.clamp(x_affine - x_affine.detach() + torch.round(x_affine), 0, qmax)
        return x_int

    def forward(self, input):
        # update activation and weight range
        if self.training:
            self.activation_observer(input)
            self.activation_scale = torch.max((self.activation_observer.max_val - self.activation_observer.min_val) / self.activation_quant_max, self.eps)
            self.activation_zp_neg = torch.round(self.activation_observer.min_val / self.activation_scale)
            self.weight_observer(self.weight)
            self.weight_scale = torch.max((self.weight_observer.max_val - self.weight_observer.min_val) / self.weight_quant_max, self.eps)
            self.weight_zp_neg = torch.round(self.weight_observer.min_val / self.weight_scale)

        # convolution parameters
        parameters = {
            'bias': None,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups
        }

        # compute conv2d using self-defined CUDA kernel
        input_int = self.quantize_v2(input, self.activation_scale, self.activation_zp_neg, self.activation_quant_max)
        weight_int = self.quantize_v2(self.weight, self.weight_scale, self.weight_zp_neg, self.weight_quant_max)

        output = self.activation_scale * self.weight_scale * (
            self.conv2d_op(input_int, weight_int, **parameters) +
            # F.conv2d(input_int, weight_int, **parameters) +
            F.conv2d(input_int, torch.full_like(weight_int, self.weight_zp_neg), **parameters) +
            F.conv2d(torch.full_like(input_int, self.activation_zp_neg), weight_int, **parameters) + 
            F.conv2d(torch.full_like(input_int, self.activation_zp_neg), torch.full_like(weight_int, self.weight_zp_neg), **parameters)
        )
        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1) # Reshape the bias to match the output shape

        return output

        
class QuantMaxPool2d(nn.MaxPool2d):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
        a_bits=8,
        q_type=0,
        qaft=False,
        ptq=False,
        percentile=0.9999,
    ):
        super(QuantMaxPool2d, self).__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.activation_quantizer = AsymmetricQuantizer(
            bits=a_bits,
            observer=MovingAverageMinMaxObserver(
                q_level="L", out_channels=None
            ),
            activation_weight_flag=1,
            qaft=qaft,
        )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.max_pool2d(
            quant_input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.return_indices,
            self.ceil_mode,
        )
        return output

        
class QuantAvgPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
        a_bits=8,
        q_type=0,
        qaft=False,
        ptq=False,
        percentile=0.9999,
    ):
        super(QuantAvgPool2d, self).__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        self.activation_quantizer = AsymmetricQuantizer(
            bits=a_bits,
            observer=MovingAverageMinMaxObserver(
                q_level="L", out_channels=None
            ),
            activation_weight_flag=1,
            qaft=qaft,
        )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        output = F.avg_pool2d(
            quant_input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )
        return output


def add_quant_op(
    module,
    a_bits=8,
    w_bits=8,
    q_type=0,
    q_level=0,
    weight_observer=0,
    bn_fuse=False,
    bn_fuse_calib=False,
    pretrained_model=False,
    qaft=False,
    ptq=False,
    percentile=0.9999,
    use_ste_gradient=False,
    discarded_columns=0
):
    for name, child in module.named_children():
        # print(f'deal with module {module.__class__.__name__}.{name}, child {child.__class__.__name__}')
        if isinstance(child, nn.Linear):
            quant_linear = QuantLinear(in_features=child.in_features, out_features=child.out_features, bias=(child.bias is not None), a_bits=a_bits, w_bits=w_bits)
            if child.bias is not None:
                quant_linear.bias.data = child.bias
            quant_linear.weight.data = child.weight
            module._modules[name] = quant_linear
        elif isinstance(child, nn.Conv2d):
            quant_conv = ApproxConv2d(in_channels=child.in_channels, out_channels=child.out_channels, kernel_size=child.kernel_size, stride=child.stride, padding=child.padding, dilation=child.dilation, groups=child.groups, bias=(child.bias is not None), padding_mode=child.padding_mode, a_bits=a_bits, w_bits=w_bits, use_ste_gradient=use_ste_gradient)
            if child.bias is not None:
                quant_conv.bias.data = child.bias
            quant_conv.weight.data = child.weight
            module._modules[name] = quant_conv
        elif isinstance(child, nn.MaxPool2d):
            quant_max_pool = QuantMaxPool2d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                a_bits=a_bits,
                q_type=q_type,
                qaft=qaft,
                ptq=ptq,
                percentile=percentile,
            )
            module._modules[name] = quant_max_pool
        elif isinstance(child, nn.AvgPool2d):
            quant_avg_pool = QuantAvgPool2d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                a_bits=a_bits,
                q_type=q_type,
                qaft=qaft,
                ptq=ptq,
                percentile=percentile,
            )
            module._modules[name] = quant_avg_pool
        # elif isinstance(child, nn.Sequential):
        else:
            add_quant_op(
                child,
                a_bits=a_bits,
                w_bits=w_bits,
                q_type=q_type,
                q_level=q_level,
                weight_observer=weight_observer,
                bn_fuse=bn_fuse,
                bn_fuse_calib=bn_fuse_calib,
                pretrained_model=pretrained_model,
                qaft=qaft,
                ptq=ptq,
                percentile=percentile,
                use_ste_gradient=use_ste_gradient,
                discarded_columns=discarded_columns
            )


def prepare(
    model,
    inplace=False,
    a_bits=8,
    w_bits=8,
    q_type=0,
    q_level=0,
    weight_observer=0,
    bn_fuse=False,
    bn_fuse_calib=False,
    # quant_inference=False,
    pretrained_model=False,
    qaft=False,
    ptq=False,
    percentile=0.9999,
    use_ste_gradient=False,
    discarded_columns=0
):
    print(f'Quantizing model with a_bits={a_bits}, w_bits={w_bits}, q_type={q_type}, q_level={q_level}, weight_observer={weight_observer}, bn_fuse={bn_fuse}, bn_fuse_calib={bn_fuse_calib}, pretrained_model={pretrained_model}, qaft={qaft}, ptq={ptq}, percentile={percentile}, use_ste_gradient={use_ste_gradient}, discarded_columns={discarded_columns}')
    if not inplace:
        model = copy.deepcopy(model)
    add_quant_op(
        model,
        a_bits=a_bits,
        w_bits=w_bits,
        q_type=q_type,
        q_level=q_level,
        weight_observer=weight_observer,
        bn_fuse=bn_fuse,
        bn_fuse_calib=bn_fuse_calib,
        pretrained_model=pretrained_model,
        qaft=qaft,
        ptq=ptq,
        percentile=percentile,
        use_ste_gradient=use_ste_gradient,
        discarded_columns=discarded_columns
    )
    return model