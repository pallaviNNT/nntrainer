// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Thummala Pallavi <t.pallavi@samsung.com>
 *
 * @file unittest_layers_fully_connected_cl.cpp
 * @date 7 June 2024
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <rmsnorm_layer_cl.h>
#include <layers_common_tests.h>

auto semantic_rms = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayerCl>,
  nntrainer::RMSNormLayerCl::type, {"epsilon=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(RMSNormGPU, LayerSemantics,
                     ::testing::Values(semantic_rms));

auto rms_basic_plain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayerCl>, {"epsilon=1"},
  "2:3:3:30","rms_norm.nnlayergolden",LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto rms_plain_skip_CG = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::RMSNormLayerCl>, {"epsilon=1"},
  "2:3:3:30","rms_norm.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
  "nchw", "fp32", "fp32");


GTEST_PARAMETER_TEST(RMSNormGPU, LayerGoldenTest,
                     ::testing::Values(rms_basic_plain, rms_plain_skip_CG));
