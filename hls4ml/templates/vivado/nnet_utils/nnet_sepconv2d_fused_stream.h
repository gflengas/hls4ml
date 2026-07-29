#ifndef NNET_SEPARABLE_CONV2D_FUSED_STREAM_H_
#define NNET_SEPARABLE_CONV2D_FUSED_STREAM_H_

#include "hls_stream.h"
#include "nnet_common.h"
#include "nnet_conv2d.h"
#include "nnet_padding.h"
#include "nnet_sepconv_stream.h"

namespace nnet {

template <class data_T, class dw_res_T, typename CONFIG_T>
bool depthwise_compute_buffer_2d_local(
    const data_T &in_elem,
    ap_shift_reg<typename data_T::value_type, CONFIG_T::in_width>
        line_buffer[MAX(CONFIG_T::filt_height - 1, 1)][CONFIG_T::n_chan],
    dw_res_T &dw_pack, typename CONFIG_T::weight_t weights[CONFIG_T::kernel_size * CONFIG_T::n_chan],
    typename CONFIG_T::bias_t biases[CONFIG_T::n_chan]) {
    #pragma HLS INLINE

    const static int lShiftX = CONFIG_T::filt_width - 1;
    const static int lShiftY = CONFIG_T::filt_height - 1;

    static int pX = 0;
    static int pY = 0;
    static int sX = 0;
    static int sY = 0;

    static typename data_T::value_type kernel_data[CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=kernel_data complete

    typename dw_res_T::value_type res_out[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=res_out complete dim=0

    nnet::shift_line_buffer<data_T, CONFIG_T>(in_elem, line_buffer, kernel_data);

    const bool output_ready =
        (sX - lShiftX) == 0 && (sY - lShiftY) == 0 && pY > lShiftY - 1 && pX > lShiftX - 1;
    if (output_ready) {
        #pragma HLS INLINE recursive
        CONFIG_T::mult_config::template kernel<typename data_T::value_type, typename dw_res_T::value_type,
                                               typename CONFIG_T::mult_config>::dense(kernel_data, res_out, weights,
                                                                                      biases);

    CastDepthwiseResult:
        for (unsigned channel = 0; channel < CONFIG_T::n_filt; channel++) {
            #pragma HLS UNROLL
            dw_pack[channel] = res_out[channel];
        }
    }

    if (pX + 1 == CONFIG_T::in_width) {
        pX = 0;
        sX = 0;
        if (pY + 1 == CONFIG_T::in_height) {
            pY = 0;
            sY = 0;
        } else {
            pY = pY + 1;
            sY = ((sY - lShiftY) == 0) ? sY - CONFIG_T::stride_height + 1 : sY + 1;
        }
    } else {
        pX = pX + 1;
        sX = ((sX - lShiftX) == 0) ? sX - CONFIG_T::stride_width + 1 : sX + 1;
    }

    return output_ready;
}

template <class dw_res_T, class res_T, typename CONFIG_T>
void pointwise_compute_local(const dw_res_T &dw_pack, res_T &res_pack,
                             typename CONFIG_T::weight_t weights[CONFIG_T::n_chan * CONFIG_T::n_filt],
                             typename CONFIG_T::bias_t biases[CONFIG_T::n_filt]) {
    #pragma HLS INLINE

    typename dw_res_T::value_type data[CONFIG_T::n_chan];
    #pragma HLS ARRAY_PARTITION variable=data complete

    typename res_T::value_type res[CONFIG_T::n_filt];
    #pragma HLS ARRAY_PARTITION variable=res complete

CopyPointwiseInput:
    for (unsigned channel = 0; channel < CONFIG_T::n_chan; channel++) {
        #pragma HLS UNROLL
        data[channel] = dw_pack[channel];
    }

    #pragma HLS INLINE recursive
    CONFIG_T::mult_config::template kernel<typename dw_res_T::value_type, typename res_T::value_type,
                                           typename CONFIG_T::mult_config>::dense(data, res, weights, biases);

CastPointwiseResult:
    for (unsigned filter = 0; filter < CONFIG_T::n_filt; filter++) {
        #pragma HLS UNROLL
        res_pack[filter] = res[filter];
    }
}

template <class data_T, class dw_res_T, class res_T, typename CONFIG_T>
void separable_conv_2d_local_cl(
    hls::stream<data_T> &data, hls::stream<res_T> &res,
    typename CONFIG_T::depthwise_config::weight_t
        depthwise_weights[CONFIG_T::depthwise_config::filt_height * CONFIG_T::depthwise_config::filt_width *
                          CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::weight_t
        pointwise_weights[CONFIG_T::pointwise_config::n_chan * CONFIG_T::pointwise_config::n_filt],
    typename CONFIG_T::depthwise_config::bias_t depthwise_biases[CONFIG_T::depthwise_config::n_chan],
    typename CONFIG_T::pointwise_config::bias_t pointwise_biases[CONFIG_T::pointwise_config::n_filt]) {

    static_assert(data_T::size == CONFIG_T::depthwise_config::n_chan,
                  "local SepConv requires one pixel per input word");
    static_assert(CONFIG_T::depthwise_config::implementation == conv_implementation::linebuffer,
                  "local SepConv requires the line-buffer implementation");
    static_assert(CONFIG_T::depthwise_config::n_filt == CONFIG_T::depthwise_config::n_chan,
                  "local SepConv requires depth multiplier 1");
    static_assert(CONFIG_T::depthwise_config::filt_height == 3 && CONFIG_T::depthwise_config::filt_width == 3,
                  "local SepConv requires a 3x3 depthwise kernel");
    static_assert(CONFIG_T::depthwise_config::stride_height == 1 && CONFIG_T::depthwise_config::stride_width == 1,
                  "local SepConv requires unit depthwise stride");
    static_assert(CONFIG_T::pointwise_config::filt_height == 1 && CONFIG_T::pointwise_config::filt_width == 1,
                  "local SepConv requires a 1x1 pointwise kernel");
    static_assert(CONFIG_T::pointwise_config::stride_height == 1 && CONFIG_T::pointwise_config::stride_width == 1,
                  "local SepConv requires unit pointwise stride");
    static_assert(CONFIG_T::depthwise_config::out_height == CONFIG_T::pointwise_config::in_height,
                  "local SepConv depthwise/pointwise height mismatch");
    static_assert(CONFIG_T::depthwise_config::out_width == CONFIG_T::pointwise_config::in_width,
                  "local SepConv depthwise/pointwise width mismatch");
    static_assert(CONFIG_T::depthwise_config::n_filt == CONFIG_T::pointwise_config::n_chan,
                  "local SepConv depthwise/pointwise channel mismatch");
    static_assert(CONFIG_T::source_height + CONFIG_T::pad_top + CONFIG_T::pad_bottom ==
                      CONFIG_T::depthwise_config::in_height,
                  "local SepConv padded height mismatch");
    static_assert(CONFIG_T::source_width + CONFIG_T::pad_left + CONFIG_T::pad_right ==
                      CONFIG_T::depthwise_config::in_width,
                  "local SepConv padded width mismatch");

    static ap_shift_reg<typename data_T::value_type, CONFIG_T::depthwise_config::in_width>
        line_buffer[MAX(CONFIG_T::depthwise_config::filt_height - 1, 1)][CONFIG_T::depthwise_config::n_chan];
    #pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2

PaddedHeight:
    for (unsigned padded_y = 0; padded_y < CONFIG_T::depthwise_config::in_height; padded_y++) {
    PaddedWidth:
        for (unsigned padded_x = 0; padded_x < CONFIG_T::depthwise_config::in_width; padded_x++) {
            #pragma HLS LOOP_FLATTEN
            #pragma HLS PIPELINE II=CONFIG_T::depthwise_config::reuse_factor

            data_T input_pack;
            PRAGMA_DATA_PACK(input_pack)
            const bool source_pixel = padded_y >= CONFIG_T::pad_top &&
                                      padded_y < CONFIG_T::pad_top + CONFIG_T::source_height &&
                                      padded_x >= CONFIG_T::pad_left &&
                                      padded_x < CONFIG_T::pad_left + CONFIG_T::source_width;
            if (source_pixel) {
                input_pack = data.read();
            } else {
            ZeroPaddingPack:
                for (unsigned channel = 0; channel < data_T::size; channel++) {
                    #pragma HLS UNROLL
                    input_pack[channel] = 0;
                }
            }

            dw_res_T dw_pack;
            PRAGMA_DATA_PACK(dw_pack)
            const bool output_ready =
                depthwise_compute_buffer_2d_local<data_T, dw_res_T, typename CONFIG_T::depthwise_config>(
                    input_pack, line_buffer, dw_pack, depthwise_weights, depthwise_biases);
            if (output_ready) {
                res_T res_pack;
                PRAGMA_DATA_PACK(res_pack)
                pointwise_compute_local<dw_res_T, res_T, typename CONFIG_T::pointwise_config>(
                    dw_pack, res_pack, pointwise_weights, pointwise_biases);
                res.write(res_pack);
            }
        }
    }
}

} // namespace nnet

#endif
