// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020
 *
 * @file   fc_layer.h
 * @date   14 May 2020
 * @brief  This is Fully Connected Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Thummala Pallavi <t.pallavi@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __RMSNORM_LAYER_CL_H__
#define __RMSNORM_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

#include <opencl_buffer.h>
#include <opencl_kernel.h>

#define CREATE_IF_EMPTY_DIMS(tensor, ...) \
  do {                                    \
    if (tensor.empty())                   \
      tensor = Tensor(__VA_ARGS__);       \
  } while (0);

namespace nntrainer {

namespace props{

/**
 * @brief RMS_NORM_GAMMA_INIT_GPU Initialization Enumeration Information
 *
 */
class RMS_NORM_GAMMA_INIT_GPU final
  : public ::nntrainer::EnumProperty<::nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a RMS_NORM_GAMMA_INIT object
   */
  RMS_NORM_GAMMA_INIT_GPU(::nntrainer::Tensor::Initializer value =
                        ::nntrainer::Tensor::Initializer::ONES) {
    set(value);
  };

  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};
};


/**
 * @class   RMSNormLayer
 * @brief   RMS Norm layer
 */
class RMSNormLayerCl : public LayerImpl {
public:
  /**
   * @brief     Constructor of RMS Norm Layer
   */
  RMSNormLayerCl();

  /**
   * @brief     Destructor of RMS Norm Layer
   */
  ~RMSNormLayerCl() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  RMSNormLayerCl(RMSNormLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RMS Norm to be moved.
   */
  RMSNormLayerCl &operator=(RMSNormLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return RMSNormLayerCl::type;
  };

  static opencl::Kernel kernel_rmsnorm;

  /**
   * @brief Process data and dimensions for dot operation used in fc_layer
   * @param[in] input Tensor
   * @param[in] weight Tensor
   * @param[in] result Tensor
   * @param[in] RunLayerContext reference
   */


  void rmsnormProcess(Tensor const &input, Tensor &result, Tensor const &gamma, const int epsilon,
                    RunLayerContext &context);


  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override {
    return true;
  }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "rmsnorm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::RMS_NORM_GAMMA_INIT_GPU, props::Epsilon>
    rmsnorm_props; /**< rmsnorm layer properties */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RMSNORM_LAYER_CL__ */

