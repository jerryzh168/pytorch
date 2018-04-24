#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/conv_op.h"

namespace caffe2 {

template <typename T>
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
  arm_compute::CLDepthwiseConvolutionLayer depth_conv_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, filter_, bias_;
};

template <typename T>
bool CLConvOp<T>::RunOnDevice() {
  auto *Xblob = OperatorBase::Inputs()[0];
  auto *filterblob = OperatorBase::Inputs()[1];
  auto *biasblob = OperatorBase::Inputs()[2];
  X_ = CLContext::getCLTensor<T>(Xblob, X_.release());
  if (first_run_) {
    filter_ = CLContext::getCLTensor<T>(filterblob);
    bias_ = CLContext::getCLTensor<T>(biasblob);
  }

  OpenCLTensor<T> *Y =
    OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();

  const int N = X_->dim32(0), H = X_->dim32(2), W = X_->dim32(3), C = X_->dim32(1);
  LOG(INFO) << "[C2DEBUG] Conv " << N << " " << H << " " << W << " " << C;
  CAFFE_ENFORCE_EQ(kernel_.size(), 2,
                   "Only 2d convolution is supported with ARM compute backend");

  CAFFE_ENFORCE(X_->ndim(), filter_->ndim());
  const int M = filter_->dim32(0);
  CAFFE_ENFORCE(filter_->dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter_->dim32(3) == kernel_w());
  LOG(ERROR) << "[C2DEBUG] group: " << group_;
  bool depthwise = group_ == C;
  if (depthwise) {
    CAFFE_ENFORCE(filter_->dim32(1) == 1);
  } else {
    CAFFE_ENFORCE(filter_->dim32(1) == C);
  }

  if (first_run_) {
    first_run_ = false;

    // resize output accordingly
    TensorCPU fakeX;
    fakeX.Resize(X_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    Y->ResizeLike(fakeY);
    LOG(INFO) << "[C2DEBUG] dims of X " << X_->dims();
    LOG(INFO) << "[C2DEBUG] dims of X(gctensor) "
      << X_->get_underlying()->info()->dimension(3) << " "
      << X_->get_underlying()->info()->dimension(2) << " "
      << X_->get_underlying()->info()->dimension(1) << " "
      << X_->get_underlying()->info()->dimension(0) << " "
    ;
    LOG(INFO) << "[C2DEBUG] dims of Y " << Y->dims();
    LOG(INFO) << "[C2DEBUG] dims of Y(gctensor) "
      << Y->get_underlying()->info()->dimension(3) << " "
      << Y->get_underlying()->info()->dimension(2) << " "
      << Y->get_underlying()->info()->dimension(1) << " "
      << Y->get_underlying()->info()->dimension(0) << " "
    ;

    if (depthwise) {
      depth_conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                            Y->get_underlying(),
                            arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
    } else {
      conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                      Y->get_underlying(),
                      arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
    }

  } else if (second_run_) {
    // Always attempt to copy the CPU to GPU on input
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, second_run_);
    bias_->lazy_allocate(biasblob, second_run_, second_run_);
    second_run_ = false;
    Y->allocate();
    if (depthwise) {
      depth_conv_.run();
    } else {
      conv_.run();
    }
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    TensorCPU fakeX;
    fakeX.Resize(X_->dims());
    TensorCPU fakeY;
    ConvPoolOpBase<CLContext>::SetOutputSize(fakeX, &fakeY, filter_->dim32(0));
    bool need_allocation = Y->ResizeLike(fakeY, true);
    if (need_allocation) {
      Y->allocate();
    }
    if (depthwise) {
      depth_conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                            Y->get_underlying(),
                            arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
      depth_conv_.run();
    } else {
      conv_.configure(X_->get_underlying(), filter_->get_underlying(), bias_->get_underlying(),
                      Y->get_underlying(),
                      arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]));
      conv_.run();
    }
 }

  return true;
}

REGISTER_CL_OPERATOR(Conv, CLConvOp<DataType>);

} // namespace caffe2
