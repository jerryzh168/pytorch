#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"
#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {

template <typename T> class CLConcatOp final : public Operator<CLContext> {
public:
  CLConcatOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CLConcatOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CLDepthConcatenateLayer concat_layer_;
  bool first_run_ = true, second_run_ = true;
  std::vector<CLContext::deleted_unique_ptr<const OpenCLTensor<T>>> inputs_;
  int channelCount_ = 0;
};


template <typename T>
bool CLConcatOp<T>::RunOnDevice() {

  CAFFE_ENFORCE(InputSize() <= 4 && InputSize() >= 2, "Number \
  of input must be between 2 and 4.");

  auto *X0blob = OperatorBase::Inputs()[0];
  if (first_run_) {
    auto X0 = CLContext::getCLTensor<T>(X0blob);
    inputs_.push_back(std::move(X0));
  } else {
    auto X0 = CLContext::getCLTensor<T>(X0blob, inputs_[0].release());
    inputs_[0] = std::move(X0);
  }

  int N = inputs_[0]->dim32(0);
  int channels = inputs_[0]->dim32(1);
  int height = inputs_[0]->dim32(2);
  int width = inputs_[0]->dim32(3);
  std::vector<const Blob*> inputsBlob;
  inputsBlob.push_back(X0blob);

  if (first_run_) {
    channelCount_ = channels;
    for (int i = 1; i < Inputs().size(); ++i) {
      auto *Xblob = OperatorBase::Inputs()[i];
      auto X = CLContext::getCLTensor<T>(Xblob);
      CAFFE_ENFORCE_EQ(N, X->dim32(0), X->dim32(0));
      CAFFE_ENFORCE_EQ(height, X->dim32(2), X->dim32(2));
      CAFFE_ENFORCE_EQ(width, X->dim32(3), X->dim32(3));
      channelCount_ += X->dim32(1);
      inputs_.push_back(std::move(X));
    }
  } else {
    channelCount_ = channels;
    for (int i = 1; i < Inputs().size(); ++i) {
      auto *Xblob = OperatorBase::Inputs()[i];
      auto X = CLContext::getCLTensor<T>(Xblob, inputs_[i].release());
      CAFFE_ENFORCE_EQ(N, X->dim32(0), X->dim32(0));
      CAFFE_ENFORCE_EQ(height, X->dim32(2), X->dim32(2));
      CAFFE_ENFORCE_EQ(width, X->dim32(3), X->dim32(3));
      channelCount_ += X->dim32(1);
      inputs_[i] = std::move(X);
    }
  }

  for (int i = 1; i < Inputs().size(); ++i) {
    auto *Xblob = OperatorBase::Inputs()[i];
    inputsBlob.push_back(Xblob);
  }
  std::vector<int64_t> output_dims = {N, channelCount_, height, width};
  OpenCLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->Resize(output_dims);
    std::vector<arm_compute::ICLTensor*> inputsTensor;
    for (int i = 0; i < inputs_.size(); ++i) {
      inputsTensor.push_back(inputs_[i]->get_underlying());
    }
    concat_layer_.configure(inputsTensor, Y->get_underlying());
  } else if (second_run_) {
    for (int i = 0; i < inputs_.size(); ++i) {
      auto* X = inputs_[i].get();
      auto* Xblob = inputsBlob[i];
      X->lazy_allocate(Xblob, second_run_, true);
    }
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    concat_layer_.run();
  } else {
    bool need_allocation = Y->Resize(output_dims);
    // Configure
    std::vector<arm_compute::ICLTensor*> inputsTensor;
    for (int i = 0; i < inputs_.size(); ++i) {
      inputsTensor.push_back(inputs_[i]->get_underlying());
    }
    concat_layer_.configure(inputsTensor, Y->get_underlying());
    // Allocate
    for (int i = 0; i < inputs_.size(); ++i) {
      auto* X = inputs_[i].get();
      auto* Xblob = inputsBlob[i];
      X->lazy_allocate(Xblob, second_run_, true);
    }
    if (need_allocation) {
      Y->allocate();
    }
    // Run
    concat_layer_.run();
  }

  return true;
}

REGISTER_CL_OPERATOR(Concat, CLConcatOp<DataType>);

} // namespace caffe2
