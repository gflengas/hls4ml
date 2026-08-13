from pathlib import Path

import numpy as np
import tensorflow as tf

import hls4ml


def _convert(model, config, output_dir, fusion_layers=None):
    kwargs = {}
    if fusion_layers is not None:
        kwargs['sepconv_fusion_layers'] = fusion_layers
    return hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(output_dir),
        project_name='fused_sepconv',
        backend='VitisUnified',
        board='zcu102',
        part='xczu9eg-ffvb1156-2-e',
        clock_period='10ns',
        clock_uncertainty='12.5%',
        io_type='io_stream',
        axi_mode='axi_stream',
        **kwargs,
    )


def test_vitisunified_io_stream_sepconv_fusion_is_opt_in_and_bit_exact(tmp_path):
    tf.keras.utils.set_random_seed(7)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 4)),
            tf.keras.layers.SeparableConv2D(
                filters=4,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='same',
                depth_multiplier=1,
                data_format='channels_last',
                use_bias=True,
                name='target_sepconv',
            ),
        ]
    )
    config = hls4ml.utils.config_from_keras_model(
        model,
        default_precision='fixed<16,6>',
        granularity='name',
        backend='VitisUnified',
    )
    config['Model']['Strategy'] = 'Latency'
    config['Model']['ReuseFactor'] = 1

    unfused = _convert(model, config, tmp_path / 'unfused')
    unfused_topology = [node.class_name for node in unfused.graph.values()]
    assert unfused_topology == ['Input', 'ZeroPadding2D', 'DepthwiseConv2D', 'PointwiseConv2D']
    pointwise_name = next(node.name for node in unfused.graph.values() if node.class_name == 'PointwiseConv2D')

    fused = _convert(model, config, tmp_path / 'fused', fusion_layers=[pointwise_name])
    assert [node.class_name for node in fused.graph.values()] == ['Input', 'FusedPaddedSeparableConv2D']
    fused.write()
    source = (Path(fused.config.get_output_dir()) / 'firmware/fused_sepconv.cpp').read_text()
    assert source.count('separable_conv_2d_local_cl') == 1
    assert 'zeropad2d_cl<' not in source
    assert 'depthwise_conv_2d_cl<' not in source
    assert 'pointwise_conv_2d_cl<' not in source
    assert 'separable_conv_2d_streams_cl' not in source

    inputs = np.random.default_rng(11).uniform(-0.5, 0.5, size=(3, 8, 8, 4))
    unfused.compile()
    fused.compile()
    unfused_result = unfused.predict(inputs)
    fused_result = fused.predict(inputs)
    np.testing.assert_array_equal(fused_result, unfused_result)
