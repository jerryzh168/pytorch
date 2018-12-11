#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

template <typename T> class CopyFromCLOp final : public Operator<CLContext> {
public:
  CopyFromCLOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws) {}
  virtual ~CopyFromCLOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  bool first_run_ = true, second_run_ = true;
  std::vector<CLContext::deleted_unique_ptr<const OpenCLTensor<T>>> inputs_;
};

template <typename T>
bool CopyFromCLOp<T>::RunOnDevice() {

  std::vector<const Blob*> inputsBlob;

  for (int i = 0; i < Inputs().size(); ++i) {
    auto *Xblob = OperatorBase::Inputs()[i];
    inputsBlob.push_back(Xblob);
  }

  if (first_run_) {
    for (int i = 0; i < Inputs().size(); ++i) {
      auto *Xblob = inputsBlob[i];
      auto X = CLContext::getCLTensor<T>(Xblob);
      inputs_.push_back(std::move(X));
    }
  } else {
    for (int i = 0; i < Inputs().size(); ++i) {
      auto *Xblob = inputsBlob[i];
      auto X = CLContext::getCLTensor<T>(Xblob, inputs_[i].release());
      inputs_[i] = std::move(X);
    }
  }

  if (first_run_) {
    first_run_ = false;
    for (int i = 0; i < Inputs().size(); ++i) {
      Output(i, inputs_[i]->dims(), at::dtype<float>().device(CPU));
    }
  } else {
    for (auto i = 0; i < Inputs().size(); ++i) {
      // Blob
      auto* Xblob = inputsBlob[i];
      // OpenCLTensor
      auto* X = inputs_[i].get();
      X->lazy_allocate(Xblob, second_run_, true);
      auto* Y = Output(i, X->dims(), at::dtype<float>().device(CPU));
      Timer timer;
      timer.Start();
      getTensorCPU(*X, *Y);
      auto millis = timer.MilliSeconds();
      //LOG(ERROR) << "[C2DEBUG] copy_op " << X->dims() << " takes " << millis << " milliseconds";
      second_run_ = false;
    }
    second_run_ = false;
  }

  return true;
}

REGISTER_CL_OPERATOR(CopyFromCL, CopyFromCLOp<DataType>);

} // namespace caffe2
