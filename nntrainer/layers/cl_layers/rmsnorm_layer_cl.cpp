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
#include <node_exporter.h>
#include <util_func.h>
#include <stdio.h>

std::string rmsnorm_cl_kernel_fp16_ =
  R"(
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    __kernel void rmsnorm_cl_fp16(__global const half* input,
                       __global half* output,
                       __global const half* gamma,
                       half epsilon,
                       int N, int C, int H, int W){
    int n = get_global_id(0);
    int c = get_global_id(1);
    int h = get_global_id(2);

    // Calculate base index for the current (n, c, h) slice
    int index_base = ((n * C + c) * H + h) * W;

    // Compute the mean square along the width dimension
    while(index_base < N*C*H*W){
        half mean_square = 0.0f;
        for (int w = 0; w < W; ++w)
        {
             half val = input[index_base + w];
             mean_square += val * val;
        }
        mean_square /= W;
        // Compute the RMS value
        half rms_value = sqrt(mean_square + epsilon);
        // Normalize inputs and scale by gamma
        for (int w = 0; w < W; ++w) {
            half input_val = input[index_base + w];
            output[index_base + w] = (input_val / rms_value) * gamma[c];
        }
        index_base += W;
}
})";

std::string rmsnorm_cl_kernel_ =
  R"(__kernel void rmsnorm_cl(__global const float* input,
                       __global float* output,
                       __global const float* gamma,
                       float epsilon,
                       int N, int C, int H, int W){
    int n = get_global_id(0);
    int c = get_global_id(1);
    int h = get_global_id(2);

    // Calculate base index for the current (n, c, h) slice
    int index_base = ((n * C + c) * H + h) * W;

    // Compute the mean square along the width dimension
    while(index_base < N*C*H*W){
    float mean_square = 0.0f;
    for (int w = 0; w < W; ++w) {
        float val = input[index_base + w];
        mean_square += val * val;
    }
    mean_square /= W;
    printf("mean_square is %f\n",mean_square);
    // Compute the RMS value
    float rms_value = sqrt(mean_square + epsilon);
    // Normalize inputs and scale by gamma
    for (int w = 0; w < W; ++w) {
        float input_val = input[index_base + w];
        output[index_base + w] = (input_val / rms_value) * gamma[c];
    }
    index_base += W;
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
  auto a = dim[0].getDim();
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
  else{
      rmsnormProcess_fp16(in,out,gamma,epsilon,context);
  }
}

opencl::Kernel RMSNormLayerCl::kernel_rmsnorm;
opencl::Kernel RMSNormLayerCl::kernel_rmsnorm_fp16;


void RMSNormLayerCl::rmsnormProcess(Tensor const &input,
                                         Tensor &result,Tensor const &gamma,const float epsilon,
                                         RunLayerContext &context){
  
  
  bool ret = false; 
  int dim1 = input.batch() * input.height() * input.width()* input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                         input.width(), input.getTensorType());
  int b=input.batch();
  int c = input.channel();
  int h = input.height();
  int w = input.width();
  do {
    ret =
      context.clCreateKernel(rmsnorm_cl_kernel_, context.LayerKernel::RMSNORM,
                             RMSNormLayerCl::kernel_rmsnorm);
    if (!ret) {
      break;
    }
    opencl::Buffer inputbuf(context.context_inst_, dim1 * sizeof(float), true,
                          nullptr);

    opencl::Buffer gammabuf(context.context_inst_, input.width() * sizeof(float), true,
                          nullptr);
    opencl::Buffer resultbuf(context.context_inst_, dim1 * sizeof(float), true, nullptr);

    const float *data = input.getData();
    float *rdata = result.getData();
    const float *gdata = gamma.getData();
    ret = inputbuf.WriteData(context.command_queue_inst_, data);
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
      4, &b, sizeof(int));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      3, &epsilon, sizeof(float));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      5, &c, sizeof(int));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      6, &h, sizeof(int));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm.SetKernelArguments(
      7, &w, sizeof(int));
    if (!ret) {
      break;
    }
    const int work_groups_count[3] = {1, 1, 1};
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

void RMSNormLayerCl::rmsnormProcess_fp16(Tensor const &input,
                                         Tensor &result,Tensor const &gamma,const float epsilon,
                                         RunLayerContext &context){


  bool ret = false;
  int dim1 = input.batch() * input.height() * input.width()* input.channel();
  CREATE_IF_EMPTY_DIMS(result, input.batch(), input.channel(), input.height(),
                         input.width(), input.getTensorType());
  int b=input.batch();
  int c = input.channel();
  int h = input.height();
  int w = input.width();
  do {
    ret =
      context.clCreateKernel(rmsnorm_cl_kernel_fp16_, context.LayerKernel::RMSNORM_FP16,
                             RMSNormLayerCl::kernel_rmsnorm_fp16);
    if (!ret) {
      break;
    }
    opencl::Buffer inputbuf(context.context_inst_, dim1 * sizeof(cl_half), true,
                          nullptr);

    opencl::Buffer gammabuf(context.context_inst_, input.width() * sizeof(cl_half), true,
                          nullptr);
    opencl::Buffer resultbuf(context.context_inst_, dim1 * sizeof(cl_half), true, nullptr);

    const __fp16 *data = input.getData<__fp16>();
    __fp16 *rdata = result.getData<__fp16>();
    const __fp16 *gdata = gamma.getData<__fp16>();
    ret = inputbuf.WriteData(context.command_queue_inst_, data);
    if (!ret) {
      break;
    }

    ret = gammabuf.WriteData(context.command_queue_inst_, gdata);
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      0, &inputbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      1, &resultbuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      2, &gammabuf, sizeof(cl_mem));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      4, &b, sizeof(int));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      3, &epsilon, sizeof(cl_half));
    if (!ret) {
      break;
    }

    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      5, &c, sizeof(int));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      6, &h, sizeof(int));
    if (!ret) {
      break;
    }
    ret = RMSNormLayerCl::kernel_rmsnorm_fp16.SetKernelArguments(
      7, &w, sizeof(int));
    if (!ret) {
      break;
    }
    const int work_groups_count[3] = {1, 1, 1};
    const int work_group_size[3] = {32, 32, 1}; // test-value

    ret = context.command_queue_inst_.DispatchCommand(
      RMSNormLayerCl::kernel_rmsnorm_fp16, work_groups_count, work_group_size);
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
  else{
      rmsnormProcess_fp16(in,out,gamma,epsilon,context);
  }
}

void RMSNormLayerCl::calcDerivative(RunLayerContext &context) {
  ml_logi("Training not supported");
}

void RMSNormLayerCl::calcGradient(RunLayerContext &context) {
  ml_logi("Training not supported");
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

