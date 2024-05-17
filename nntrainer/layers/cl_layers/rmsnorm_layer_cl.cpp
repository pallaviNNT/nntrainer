// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file        fc_layer_cl.cpp
 * @date        7 May 2020
 * @brief       This is Fully Connected Layer Class for Neural Network with OpenCl
 * implementation
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Debadri Samaddar <s.debadri@samsung.com>
 * @bug         No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <rmsnorm_layer_cl.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

std::string rmsnorm_cl_kernel_ =
  R"(__kernel void rms_norm(
    __global const float* input,  // Input tensor
    __global float* output,       // Output tensor
    __global const float* gamma,  // Scale factor tensor
    const float epsilon,          // Small constant to prevent division by zero
    const int size                // Size of the input tensor
) {
    // Get the index of the current work-item
    int gid = get_global_id(0);

    // Step 1: Compute the mean square of the input
    float mean_square = 0.0f;
    for (int i = 0; i < size; i++) {
        float val = input[gid * size + i];
        mean_square += val * val;
    }
    mean_square /= size;

    // Step 2: Compute the root mean square
    float rms = sqrt(mean_square + epsilon);

    // Step 3: Normalize the input and apply gamma
    for (int i = 0; i < size; i++) {
        int index = gid * size + i;
        output[index] = (input[index] / rms) * gamma[i];
    }
})";

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum RMSParams { gamma };

RMSNormLayerCl::RMSNormLayerCl() :
  LayerImpl() {
  wt_idx.fill(0);
}

void RMSNormLayerCl::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();

  context.setOutputDimensions(dim);

  auto &rmsparams_gamma = std::get<props::RMS_NORM_GAMMA_INIT_GPU>(rmsnorm_props);

  TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, rmsparams_gamma, WeightRegularizer::NONE, 1.0f, 0.0f,
    "gamma", false);
}

void RMSNormLayerCl::forwarding(RunLayerContext &context,
                              bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();

  if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
	  rmsnormProcess(in,out,gamma,epsilon,context);
  }
  else
	  throw std::runtime_error("Only FP32 is supported");
}

opencl::Kernel RMSNormLayerCl::kernel_rmsnorm;


void RMSNormLayerCl::rmsnormProcess(Tensor const &input,
                                         Tensor &result,Tensor const &gamma,const int epsilon,
                                         RunLayerContext &context){
  
  
  bool ret = false; 
  int dim1 = input.batch() * input.height() * input.width()* input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                         input.width(), input.getTensorType());
  do {
    ret =
      context.clCreateKernel(rmsnorm_cl_kernel_, context.LayerKernel::RMSNORM,
                             RMSNormLayerCl::kernel_rmsnorm);
    if (!ret) {
      break;
    }

    opencl::Buffer inputbuf(context.context_inst_, dim1 * sizeof(float), true,
                          nullptr);

    opencl::Buffer gammabuf(context.context_inst_, dim1 * sizeof(float), true,
                          nullptr);
    opencl::Buffer resultbuf(context.context_inst_, sizeof(float), true, nullptr);


    const float *data = input.getData();
    float *rdata = result.getData();
    const float *gdata = gamma.getData();
    ret = inputbuf.WriteData(context.command_queue_inst_, data);
    if (!ret) {
      break;
    }

    ret = resultbuf.WriteData(context.command_queue_inst_, rdata);
    if (!ret) {
      break;
    }

    ret = gammabuf.WriteData(context.command_queue_inst_, gdata);
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      0, &inputbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      1, &resultbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    

    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      2, &gammabuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      3, &dim1, sizeof(int));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      4, &epsilon, sizeof(int));
    if (!ret) {
      break;
    }

    const int work_groups_count[3] = {(int)dim1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    ret = context.command_queue_inst_.DispatchCommand(
      RMSNormLayerCl::kernel_rmsnorm, work_groups_count, work_group_size);
    if (!ret) {
      break;
    }

    ret = resultbuf.ReadData(context.command_queue_inst_, rdata);
    if (!ret) {
      break;
    }

  } while (false);

}

void RMSNormLayerCl::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  in_step_dim.height(to - from);
  out_step_dim.height(to - from);

  Tensor in_step = in.getSharedDataTensor(in_step_dim, 0, true);
  Tensor out_step = out.getSharedDataTensor(out_step_dim, 0, true);

  auto &epsilon = std::get<props::Epsilon>(rmsnorm_props).get();

  if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
          rmsnormProcess(in,out,gamma,epsilon,context);
  }
  else
          throw std::runtime_error("Only FP32 is supported");
}

void RMSNormLayerCl::calcDerivative(RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

void RMSNormLayerCl::calcGradient(RunLayerContext &context) {
  // std::throw_with_nested(std::runtime_error("Training is not supported
  // yet."));
}

void RMSNormLayerCl::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(rmsnorm_props, method, this);
}

void RMSNormLayerCl::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, rmsnorm_props);
  LayerImpl::setProperty(remain_props);
}

}

