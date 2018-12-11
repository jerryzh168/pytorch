#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "arm_compute/runtime/CL/CLSubTensor.h"
#include "caffe2/operators/conv_op.h"

namespace caffe2 {

struct EmptyActivation {
  static string act() {
    return "";
  }
};

struct ReluActivation {
  static string act() {
    return "relu";
  }
};

struct SigmoidActivation {
  static string act() {
    return "sigmoid";
  }
};
template <typename T, typename Activation>
class CLConvOp final : public ConvPoolOpBase<CLContext> {
 public:
  USE_CONV_POOL_BASE_FUNCTIONS(CLContext);
  CLConvOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<CLContext>(operator_def, ws) {
    // Since this is the default convolution implementation, we will
    // use CAFFE_ENFORCE instead of OPERATOR_NEEDS_FEATURE.
    CAFFE_ENFORCE(
        group_ == 1 || order_ == StorageOrder::NCHW,
        "Group convolution only supports NCHW order right now.");
  }
  ~CLConvOp() {}

  bool RunOnDevice() override;
private:
  arm_compute::CLConvolutionLayer conv_;
  arm_compute::CLConvolutionLayer gconv_;
  arm_compute::CLDepthwiseConvolutionLayer3x3 depth_conv_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, filter_, bias_;
};

template <typename T, typename Activation>
bool CLConvOp<T, Activation>::RunOnDevice() {
  auto *Xblob = OperatorBase::Inputs()[0];
  auto *filterblob = OperatorBase::Inputs()[1];
  auto *biasblob = OperatorBase::Inputs()[2];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());
  filter_ = CLContext::getCLTensor<T>(filterblob, filter_.release());
  bias_ = CLContext::getCLTensor<T>(biasblob, bias_.release());

  OpenCLTensor<T> *Y =
    OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

  LOG(ERROR) << "[C2DEBUG] : 1";
  const int N = X_->dim32(0), H = X_->dim32(2), W = X_->dim32(3), C = X_->dim32(1);
  CAFFE_ENFORCE_EQ(kernel_.size(), 2,
                   "Only 2d convolution is supported with ARM compute backend");

  LOG(ERROR) << "[C2DEBUG] : 2";
  CAFFE_ENFORCE(X_->ndim() == filter_->ndim());
  const int M = filter_->dim32(0);
  CAFFE_ENFORCE(filter_->dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter_->dim32(3) == kernel_w());
  bool depthwise = group_ == C;
  bool grouped = group_ > 1;
  CAFFE_ENFORCE(C % group_ == 0);
  CAFFE_ENFORCE(M % group_ == 0);
  CAFFE_ENFORCE(filter_->dim32(1) == C / group_);

  LOG(ERROR) << "[C2DEBUG] : 3";

  if (first_run_) {
    first_run_ = false;

    // resize output accordingly
    Tensor fakeX(X_->dims(), CPU);
    Tensor fakeY(CPU);
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    Y->ResizeLike(fakeY);

    if (depthwise) {
      depth_conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                            Y->get_underlying(),
                            arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
    } else if (grouped) {
      gconv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(), Y->get_underlying(),
                       arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(),
                       arm_compute::Size2D(1, 1), arm_compute::ActivationLayerInfo(), false, group_);
    } else {
      arm_compute::ActivationLayerInfo act_info;
      string activation = Activation::act();
      if (activation == "relu") {
        act_info = arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
      } else if (activation == "sigmoid") {
        act_info = arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
      } else {
        act_info = arm_compute::ActivationLayerInfo();
      }
      conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(), Y->get_underlying(),
                      arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(),
                      arm_compute::Size2D(1, 1), act_info);
    }
    LOG(ERROR) << "[C2DEBUG] : 4";
  } else if (second_run_) {
    // Always attempt to copy the CPU to GPU on input
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, second_run_);
    bias_->lazy_allocate(biasblob, second_run_, second_run_);
    second_run_ = false;
    Y->allocate();
    if (depthwise) {
      depth_conv_.run();
    } else if (grouped) {
      gconv_.run();
    } else {
      conv_.run();
    }
  } else {
    Tensor fakeX(X_->dims(), CPU);
    Tensor fakeY(CPU);
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    LOG(ERROR) << "[C2DEBUG] after SetOutputSize";
    bool need_allocation = Y->ResizeLike(fakeY, true);
    // Configure
    if (depthwise) {
      depth_conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                            Y->get_underlying(),
                            arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
    } else if (grouped) {
      gconv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(), Y->get_underlying(),
                       arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(false, 0, 0, 0, true /* retain weights from first run */),
                       arm_compute::Size2D(1, 1), arm_compute::ActivationLayerInfo(), false, group_);
    } else {
      arm_compute::ActivationLayerInfo act_info;
      string activation = Activation::act();
      if (activation == "relu") {
        act_info = arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::RELU);
      } else if (activation == "sigmoid") {
        act_info = arm_compute::ActivationLayerInfo(arm_compute::ActivationLayerInfo::ActivationFunction::LOGISTIC);
      } else {
        act_info = arm_compute::ActivationLayerInfo();
      }
      conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(), Y->get_underlying(),
                      arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), arm_compute::WeightsInfo(false, 0, 0, 0, true /* retain weights from first run */),
                      arm_compute::Size2D(1, 1), act_info);
    }
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, true);
    bias_->lazy_allocate(biasblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    if (depthwise) {
      depth_conv_.run();
    } else if (grouped) {
      gconv_.run();
    } else {
      conv_.run();
    }
 }
  LOG(ERROR) << "[C2DEBUG] : 5";
  return true;
}

REGISTER_CL_OPERATOR(Conv, CLConvOp<DataType, EmptyActivation>);
REGISTER_CL_OPERATOR(ConvRelu, CLConvOp<DataType, ReluActivation>);
REGISTER_CL_OPERATOR(ConvSigmoid, CLConvOp<DataType, SigmoidActivation>);

} // namespace caffe2
