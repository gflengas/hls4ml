from collections import OrderedDict

from hls4ml.backends.fpga.fpga_layers import PointwiseConv2D
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.backends.vivado.passes.convolution_templates import Conv2DConfigTemplate
from hls4ml.backends.vivado.passes.pointwise import PointwiseConv2DConfigTemplate
from hls4ml.backends.vivado.passes.reshaping_templates import ZeroPaddingConfigTemplate
from hls4ml.model.layers import DepthwiseConv2D, Layer, ZeroPadding2D, register_layer
from hls4ml.model.optimizer import OptimizerPass


class FusedPaddedSeparableConv2D(Layer):
    """VitisUnified-only graph node for the FIFO-free local SepConv kernel."""

    def initialize(self):
        # The fusion pass transfers the finalized pointwise output variable and
        # all four weight variables before constructing this node.
        pass


def _fusion_allowlist(model):
    backend_config = model.config.get_config_value('VitisUnifiedConfig', {})
    fusion_config = backend_config.get('SepConvFusion', {})
    if not fusion_config.get('Enabled', False):
        return ()
    layers = fusion_config.get('Layers')
    if not isinstance(layers, (list, tuple)) or not all(isinstance(name, str) for name in layers):
        raise ValueError('VitisUnifiedConfig.SepConvFusion.Layers must be an explicit list of HLS layer names')
    return tuple(layers)


def _require(condition, message):
    if not condition:
        raise ValueError(f'VitisUnified SepConv fusion rejected: {message}')


class FusePaddedDepthwisePointwise2D(OptimizerPass):
    """Fuse only the supported VitisUnified io_stream padded SepConv pattern."""

    def match(self, node):
        model = node.model
        return (
            model.config.backend.name == 'VitisUnified'
            and model.config.get_config_value('IOType') == 'io_stream'
            and isinstance(node, PointwiseConv2D)
            and node.name in _fusion_allowlist(model)
        )

    def transform(self, model, pointwise):
        depthwise = pointwise.get_input_node()
        _require(isinstance(depthwise, DepthwiseConv2D), f'{pointwise.name} is not fed by DepthwiseConv2D')
        padding = depthwise.get_input_node()
        _require(isinstance(padding, ZeroPadding2D), f'{depthwise.name} is not fed by ZeroPadding2D')

        _require(depthwise.get_output_nodes() == [pointwise], f'{depthwise.name} output is not exclusive')
        _require(padding.get_output_nodes() == [depthwise], f'{padding.name} output is not exclusive')
        _require(pointwise.outputs[0] not in depthwise.outputs, 'pointwise output aliases the depthwise output')

        for layer in (padding, depthwise, pointwise):
            _require(layer.get_attr('data_format') == 'channels_last', f'{layer.name} is not channels_last')

        input_type = padding.get_input_variable().type
        output_type = pointwise.get_output_variable().type
        _require(getattr(input_type, 'n_pack', None) == 1, 'input is not one pixel per stream word')
        _require(getattr(input_type, 'n_elem', None) == depthwise.get_attr('n_chan'), 'input pack/channel mismatch')
        _require(getattr(output_type, 'n_pack', None) == 1, 'output is not one pixel per stream word')
        _require(getattr(output_type, 'n_elem', None) == pointwise.get_attr('n_filt'), 'output pack/filter mismatch')

        _require(depthwise.get_attr('implementation') == 'linebuffer', 'depthwise implementation is not linebuffer')
        _require(depthwise.get_attr('strategy') == 'latency', 'depthwise strategy is not latency')
        _require(pointwise.get_attr('strategy') == 'latency', 'pointwise strategy is not latency')
        _require(depthwise.get_attr('parallelization_factor') == 1, 'depthwise parallelization factor is not 1')
        _require(pointwise.get_attr('parallelization_factor') == 1, 'pointwise parallelization factor is not 1')
        _require(
            (depthwise.get_attr('filt_height'), depthwise.get_attr('filt_width')) == (3, 3),
            'depthwise kernel is not 3x3',
        )
        _require(
            (pointwise.get_attr('filt_height'), pointwise.get_attr('filt_width')) == (1, 1),
            'pointwise kernel is not 1x1',
        )
        _require(
            (depthwise.get_attr('stride_height'), depthwise.get_attr('stride_width')) == (1, 1),
            'depthwise stride is not 1x1',
        )
        _require(
            (pointwise.get_attr('stride_height'), pointwise.get_attr('stride_width')) == (1, 1),
            'pointwise stride is not 1x1',
        )
        _require(
            (depthwise.get_attr('dilation_height', 1), depthwise.get_attr('dilation_width', 1)) == (1, 1),
            'depthwise dilation is not 1x1',
        )
        _require(
            (pointwise.get_attr('dilation_height', 1), pointwise.get_attr('dilation_width', 1)) == (1, 1),
            'pointwise dilation is not 1x1',
        )
        _require(depthwise.get_attr('n_filt') == depthwise.get_attr('n_chan'), 'depth multiplier is not 1')
        _require(depthwise.get_attr('n_filt') == pointwise.get_attr('n_chan'), 'child channel counts differ')
        _require(depthwise.get_attr('out_height') == pointwise.get_attr('in_height'), 'child heights differ')
        _require(depthwise.get_attr('out_width') == pointwise.get_attr('in_width'), 'child widths differ')
        _require(padding.get_attr('out_height') == depthwise.get_attr('in_height'), 'padding/depthwise heights differ')
        _require(padding.get_attr('out_width') == depthwise.get_attr('in_width'), 'padding/depthwise widths differ')
        _require(padding.get_attr('n_chan') == depthwise.get_attr('n_chan'), 'padding/depthwise channels differ')
        _require(
            any(padding.get_attr(name) != 0 for name in ('pad_top', 'pad_bottom', 'pad_left', 'pad_right')),
            'padding is empty',
        )
        _require(
            all(depthwise.get_attr(name) == 0 for name in ('pad_top', 'pad_bottom', 'pad_left', 'pad_right')),
            'depthwise layer still owns padding',
        )
        _require(
            all(pointwise.get_attr(name) == 0 for name in ('pad_top', 'pad_bottom', 'pad_left', 'pad_right')),
            'pointwise layer has padding',
        )

        padding_config_cpp = ZeroPaddingConfigTemplate().format(padding)
        depthwise_config_cpp = Conv2DConfigTemplate().format(depthwise)
        pointwise_config_cpp = PointwiseConv2DConfigTemplate().format(pointwise)
        output_name = pointwise.outputs[0]
        attributes = {
            output_name: pointwise.get_output_variable(),
            'padding_config_cpp': padding_config_cpp,
            'depthwise_config_cpp': depthwise_config_cpp,
            'pointwise_config_cpp': pointwise_config_cpp,
            'padding_config_index': padding.index,
            'depthwise_config_index': depthwise.index,
            'pointwise_config_index': pointwise.index,
            'padding_layer_name': padding.name,
            'depthwise_layer_name': depthwise.name,
            'pointwise_layer_name': pointwise.name,
            'depthwise_weight': depthwise.get_weights('weight'),
            'depthwise_bias': depthwise.get_weights('bias'),
            'pointwise_weight': pointwise.get_weights('weight'),
            'pointwise_bias': pointwise.get_weights('bias'),
            'depthwise_result_t': depthwise.get_output_variable().type,
            'depthwise_accum_t': depthwise.get_attr('accum_t'),
            'pointwise_accum_t': pointwise.get_attr('accum_t'),
            'source_height': padding.get_attr('in_height'),
            'source_width': padding.get_attr('in_width'),
            'pad_top': padding.get_attr('pad_top'),
            'pad_bottom': padding.get_attr('pad_bottom'),
            'pad_left': padding.get_attr('pad_left'),
            'pad_right': padding.get_attr('pad_right'),
            'pointwise_multiplier_limit_floor': 16,
        }
        fused = model.make_node(
            'FusedPaddedSeparableConv2D',
            pointwise.name,
            attributes,
            padding.inputs.copy(),
            outputs=pointwise.outputs.copy(),
        )

        removed_outputs = set(padding.outputs + depthwise.outputs)
        model.replace_node(pointwise, fused)
        model.graph = OrderedDict(
            (name, layer) for name, layer in model.graph.items() if layer not in (padding, depthwise)
        )
        for output in removed_outputs:
            model.output_vars.pop(output, None)
        return True


class FusedPaddedSeparableConv2DConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(FusedPaddedSeparableConv2D)

    def format(self, node):
        child_configs = node.get_attr('padding_config_cpp') + node.get_attr('depthwise_config_cpp') + node.get_attr(
            'pointwise_config_cpp'
        )
        pointwise_index = node.get_attr('pointwise_config_index')
        master_config = f'''struct config{node.index}_pointwise_mult : config{pointwise_index}_mult {{
    static const unsigned multiplier_limit =
        config{pointwise_index}_mult::multiplier_limit < {node.get_attr('pointwise_multiplier_limit_floor')}
            ? {node.get_attr('pointwise_multiplier_limit_floor')}
            : config{pointwise_index}_mult::multiplier_limit;
}};

struct config{node.index}_pointwise : config{pointwise_index} {{
    typedef config{node.index}_pointwise_mult mult_config;
}};

struct config{node.index} {{
    using depthwise_config = config{node.get_attr('depthwise_config_index')};
    using pointwise_config = config{node.index}_pointwise;
    static const unsigned source_height = {node.get_attr('source_height')};
    static const unsigned source_width = {node.get_attr('source_width')};
    static const unsigned pad_top = {node.get_attr('pad_top')};
    static const unsigned pad_bottom = {node.get_attr('pad_bottom')};
    static const unsigned pad_left = {node.get_attr('pad_left')};
    static const unsigned pad_right = {node.get_attr('pad_right')};
}};\n'''
        return child_configs + master_config


class FusedPaddedSeparableConv2DFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(
            FusedPaddedSeparableConv2D,
            include_header=['nnet_utils/nnet_sepconv2d_fused_stream.h'],
        )

    def format(self, node):
        params = self._default_function_params(node)
        return (
            f'nnet::separable_conv_2d_local_cl<'
            f"{params['input_t']}, {node.get_attr('depthwise_result_t').name}, "
            f"{params['output_t']}, {params['config']}>("
            f"{params['input']}, {params['output']}, "
            f"{node.get_weights('depthwise_weight').name}, {node.get_weights('pointwise_weight').name}, "
            f"{node.get_weights('depthwise_bias').name}, {node.get_weights('pointwise_bias').name});"
        )


def register_sepconv_fusion(backend):
    register_layer('FusedPaddedSeparableConv2D', FusedPaddedSeparableConv2D)
    backend.register_pass('fuse_padded_depthwise_pointwise_2d', FusePaddedDepthwisePointwise2D)
    backend.register_template(FusedPaddedSeparableConv2DConfigTemplate)
    backend.register_template(FusedPaddedSeparableConv2DFunctionTemplate)
