#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/core/context.h"
#include "caffe2/operators/conv_op_shared.h"
#include "caffe2/operators/conv_transpose_op.h"
#include "caffe2/operators/conv_transpose_unpool_op_base.h"

namespace caffe2 {

template <typename T>
class CLConvTransposeOp final : public ConvTransposeUnpoolBase<CLContext> {
 public:
  USE_CONV_TRANSPOSE_UNPOOL_BASE_FUNCTIONS(CLContext);
  CLConvTransposeOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvTransposeUnpoolBase<CLContext>(operator_def, ws) {
  }
  ~CLConvTransposeOp() {}

  bool RunOnDevice() override;
private:
  arm_compute::CLDeconvolutionLayer conv_trans_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> X_, filter_, bias_;
  OpenCLTensor<T> refilter_;
};

template <typename T>
bool CLConvTransposeOp<T>::RunOnDevice() {
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
  LOG(INFO) << "[C2DEBUG] ConvTranspose " << N << " " << H << " " << W << " " << C;
  CAFFE_ENFORCE_EQ(kernel_.size(), 2,
                   "Only 2d convolution is supported with ARM compute backend");

  CAFFE_ENFORCE(X_->ndim(), filter_->ndim());
  const int input_channels = filter_->dim32(0);
  const int output_channels = filter_->dim32(1);
  CAFFE_ENFORCE(filter_->dim32(2) == kernel_h());
  CAFFE_ENFORCE(filter_->dim32(3) == kernel_w());
  CAFFE_ENFORCE(input_channels == C);
  CAFFE_ENFORCE(bias_->ndim(), 1);
  CAFFE_ENFORCE(bias_->dim32(0), output_channels);

  // TODO: remove filter reshaping at some point
  auto dims = filter_->dims();
  dims[0] = output_channels;
  dims[1] = input_channels;
  refilter_.Resize(dims);

  if (first_run_) {
    first_run_ = false;

    // resize output accordingly
    Tensor fakeX(X_->dims(), CPU);
    Tensor fakeY(CPU);
    ConvTransposeUnpoolBase<CLContext>::SetOutputSize(fakeX, &fakeY, output_channels);
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

    conv_trans_.configure(
        X_->get_underlying(), refilter_.get_underlying(), bias_->get_underlying(),
        Y->get_underlying(),
        arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), 0, 0);

  } else if (second_run_) {
    // Always attempt to copy the CPU to GPU on input
    X_->lazy_allocate(Xblob, second_run_, true);
    filter_->lazy_allocate(filterblob, second_run_, second_run_);
    refilter_.allocate();
    bias_->lazy_allocate(biasblob, second_run_, second_run_);
    second_run_ = false;
    Y->allocate();

    // TODO: remove filter reshaping at some point
    // Reshape filter_ from [IFM, OFM, H, W] to [OFM, IFM, H, W]
    const int kH = kernel_h();
    const int kW = kernel_w();
    auto* filter_data = filter_->map();
    auto* refilter_data = refilter_.map();
    for (auto ic = 0; ic < input_channels; ++ic) {
      for (auto oc = 0; oc < output_channels; ++oc) {
        for (auto kh = 0; kh < kH; ++kh) {
          for (auto kw = 0; kw < kW; ++kw) {
            const auto inputIdx = ic + oc * input_channels + kh * input_channels * output_channels + kw * input_channels * output_channels * kH;
            const auto outputIdx = oc + ic * output_channels + kh * output_channels * input_channels + kw * output_channels * input_channels * kH;
            DCHECK_LT(inputIdx, filter_->size());
            DCHECK_LT(outputIdx, filter_->size());
            refilter_data[outputIdx] = filter_data[inputIdx];
          }
        }
      }
    }
    refilter_.unmap();
    filter_->unmap();

    conv_trans_.run();
  } else {
    Tensor fakeX(X_->dims(), CPU);
    Tensor fakeY(CPU);
    ConvTransposeUnpoolBase<CLContext>::SetOutputSize(fakeX, &fakeY, output_channels);
    bool need_allocation = Y->ResizeLike(fakeY, true);
    // Configure
    conv_trans_.configure(
                    X_->get_underlying(), refilter_.get_underlying(), bias_->get_underlying(),
                    Y->get_underlying(),
                    arm_compute::PadStrideInfo(stride_[0], stride_[1], pads_[0], pads_[1]), 0, 0,
                    arm_compute::WeightsInfo(false, 0, 0, 0, true /* retain weights from previous run */));
    // Allocate
    X_->lazy_allocate(Xblob, second_run_, true);
    if (need_allocation) {
      Y->allocate();
    }
    // TODO: remove filter reshaping at some point
    // Reshape filter_ from [IFM, OFM, H, W] to [OFM, IFM, H, W]
    const int kH = kernel_h();
    const int kW = kernel_w();
    auto* filter_data = filter_->map();
    auto* refilter_data = refilter_.map();
    for (auto ic = 0; ic < input_channels; ++ic) {
      for (auto oc = 0; oc < output_channels; ++oc) {
        for (auto kh = 0; kh < kH; ++kh) {
          for (auto kw = 0; kw < kW; ++kw) {
            const auto inputIdx = ic + oc * input_channels + kh * input_channels * output_channels + kw * input_channels * output_channels * kH;
            const auto outputIdx = oc + ic * output_channels + kh * output_channels * input_channels + kw * output_channels * input_channels * kH;
            DCHECK_LT(inputIdx, filter_->size());
            DCHECK_LT(outputIdx, filter_->size());
            refilter_data[outputIdx] = filter_data[inputIdx];
          }
        }
      }
    }
    refilter_.unmap();
    filter_->unmap();

    // Run
    conv_trans_.run();
 }

  return true;
}

REGISTER_CL_OPERATOR(ConvTranspose, CLConvTransposeOp<DataType>);

} // namespace caffe2
