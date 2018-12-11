#include "caffe2/mobile/contrib/opencl/core/context.h"
#include "caffe2/mobile/contrib/opencl/core/operator.h"

#include "caffe2/operators/box_with_nms_limit_op.h"

namespace caffe2 {

template <typename T> class CPPBoxWithNMSLimitOp final : public Operator<CLContext> {
public:
  CPPBoxWithNMSLimitOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<CLContext>(operator_def, ws),
        score_thres_(OperatorBase::GetSingleArgument<float>("score_thresh", 0.05)),
        nms_thres_(OperatorBase::GetSingleArgument<float>("nms", 0.3)),
        detections_per_im_(OperatorBase::GetSingleArgument<int>("detections_per_im", 100)),
        soft_nms_enabled_(OperatorBase::GetSingleArgument<bool>("soft_nms_enabled", false)),
        soft_nms_method_str_(OperatorBase::GetSingleArgument<std::string>("soft_nms_method", "linear")),
        soft_nms_sigma_(OperatorBase::GetSingleArgument<float>("soft_nms_sigma", 0.5)),
        soft_nms_min_score_thres_(OperatorBase::GetSingleArgument<float>("soft_nms_min_score_thres", 0.001)) {
    CAFFE_ENFORCE(
        soft_nms_method_str_ == "linear" || soft_nms_method_str_ == "gaussian",
        "Unexpected soft_nms_method");
    soft_nms_method_ = (soft_nms_method_str_ == "linear") ? arm_compute::NMSType::LINEAR : arm_compute::NMSType::GAUSSIAN;
    info_ = arm_compute::BoxNMSLimitInfo(score_thres_, nms_thres_, detections_per_im_,
                                         soft_nms_enabled_, soft_nms_method_,
                                         soft_nms_sigma_, soft_nms_min_score_thres_);
  }

  virtual ~CPPBoxWithNMSLimitOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(CLContext);
  bool RunOnDevice() override;
private:
  arm_compute::CPPBoxWithNonMaximaSuppressionLimit box_with_nms_limit_layer_;
  bool first_run_ = true, second_run_ = true;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> tscores_;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> tboxes_;
  CLContext::deleted_unique_ptr<const OpenCLTensor<T>> batch_splits_;
protected:
  // TEST.SCORE_THRESH
  float score_thres_ = 0.05;
  // TEST.NMS
  float nms_thres_ = 0.3;
  // TEST.DETECTIONS_PER_IM
  int detections_per_im_ = 100;
  // TEST.SOFT_NMS.ENABLED
  bool soft_nms_enabled_ = false;
  // TEST.SOFT_NMS.METHOD
  std::string soft_nms_method_str_ = "linear";
  arm_compute::NMSType soft_nms_method_ = arm_compute::NMSType::LINEAR;
  // TEST.SOFT_NMS.SIGMA
  float soft_nms_sigma_ = 0.5;
  // Lower-bound on updated scores to discard boxes
  float soft_nms_min_score_thres_ = 0.001;
  arm_compute::BoxNMSLimitInfo info_;
};

template <typename T>
bool CPPBoxWithNMSLimitOp<T>::RunOnDevice() {

  auto *tscoresblob = OperatorBase::Inputs()[0];
  auto *tboxesblob  = OperatorBase::Inputs()[1];
  tscores_          = CLContext::getCLTensor<T>(tscoresblob, tscores_.release());
  tboxes_           = CLContext::getCLTensor<T>(tboxesblob, tboxes_.release());

  // tscores_: (num_boxes, num_classes), 0 for background
  if (tscores_->ndim() == 4) {
    CAFFE_ENFORCE_EQ(tscores_->dim32(2), 1, tscores_->dim32(2));
    CAFFE_ENFORCE_EQ(tscores_->dim32(3), 1, tscores_->dim32(3));
  } else {
    CAFFE_ENFORCE_EQ(tscores_->ndim(), 2, tscores_->ndim());
  }
  // tboxes: (num_boxes, num_classes * 4)
  if (tboxes_->ndim() == 4) {
    CAFFE_ENFORCE_EQ(tboxes_->dim32(2), 1, tboxes_->dim32(2));
    CAFFE_ENFORCE_EQ(tboxes_->dim32(3), 1, tboxes_->dim32(3));
  } else {
    CAFFE_ENFORCE_EQ(tboxes_->ndim(), 2, tboxes_->ndim());
  }

  int N = tscores_->dim32(0);
  int num_classes = tscores_->dim32(1);

  CAFFE_ENFORCE_EQ(N, tboxes_->dim32(0));
  CAFFE_ENFORCE_EQ(num_classes * 4, tboxes_->dim32(1));

  batch_splits_  = nullptr;
  const Blob *batchsplitsblob = nullptr;
  int batch_size = 1;
  if (InputSize() > 2) {
    // tscores and tboxes have items from multiple images in a batch. Get the
    // corresponding batch splits from input.
    batchsplitsblob = OperatorBase::Inputs()[2];
    batch_splits_   = CLContext::getCLTensor<T>(batchsplitsblob, batch_splits_.release());
    CAFFE_ENFORCE_EQ(batch_splits_->ndim(), 1);
    batch_size            = batch_splits_->dim32(0);
  }

  const int max_total_keeps = (detections_per_im_ > 0) ? detections_per_im_ : num_classes * N;

  OpenCLTensor<T> *out_scores       = OperatorBase::Outputs()[0]->template GetMutable<OpenCLTensor<T>>();
  OpenCLTensor<T> *out_boxes        = OperatorBase::Outputs()[1]->template GetMutable<OpenCLTensor<T>>();
  OpenCLTensor<T> *out_classes      = OperatorBase::Outputs()[2]->template GetMutable<OpenCLTensor<T>>();
  OpenCLTensor<T> *out_batch_splits = nullptr;
  if (OutputSize() > 3) {
    out_batch_splits = OperatorBase::Outputs()[3]->template GetMutable<OpenCLTensor<T>>();
  }
  OpenCLTensor<T> *out_keeps      = nullptr;
  OpenCLTensor<T> *out_keeps_size = nullptr;
  if (OutputSize() > 4) {
    out_keeps      = OperatorBase::Outputs()[4]->template GetMutable<OpenCLTensor<T>>();
    out_keeps_size = OperatorBase::Outputs()[5]->template GetMutable<OpenCLTensor<T>>();
  }

  if (first_run_) {
    first_run_ = false;
    out_scores->Resize(max_total_keeps);
    out_boxes->Resize(max_total_keeps, 4);
    out_classes->Resize(max_total_keeps);
    if (out_batch_splits) {
      out_batch_splits->Resize(batch_size);
    }
    if (out_keeps) {
      out_keeps->Resize(detections_per_im_);
      out_keeps_size->Resize(batch_size, num_classes);
    }
    // Configure
    box_with_nms_limit_layer_.configure(tscores_->get_underlying(), tboxes_->get_underlying(),
                                        (batch_splits_) ? batch_splits_->get_underlying() : nullptr,
                                        out_scores->get_underlying(), out_boxes->get_underlying(), out_classes->get_underlying(),
                                        (out_batch_splits) ? out_batch_splits->get_underlying() : nullptr,
                                        (out_keeps) ? out_keeps->get_underlying() : nullptr,
                                        (out_keeps_size) ? out_keeps_size->get_underlying() : nullptr,
                                        info_);
  } else if (second_run_) {
    tscores_->lazy_allocate(tscoresblob, second_run_, true);
    tboxes_->lazy_allocate(tboxesblob, second_run_, true);
    if (batch_splits_) {
      batch_splits_->lazy_allocate(batchsplitsblob, second_run_, true);
    }
    second_run_ = false;
    // Allocate
    out_scores->allocate();
    out_boxes->allocate();
    out_classes->allocate();
    if (out_batch_splits) {
      out_batch_splits->allocate();
    }
    if (out_keeps) {
      out_keeps->allocate();
      out_keeps_size->allocate();
    }
    // Since this is a CPP kernel/function we need to map and unmap tensors
    tscores_->get_underlying()->map(true);
    tboxes_->get_underlying()->map(true);
    if (batch_splits_) {
      batch_splits_->get_underlying()->map(true);
    }
    out_scores->get_underlying()->map(true);
    out_boxes->get_underlying()->map(true);
    out_classes->get_underlying()->map(true);
    if (out_batch_splits) {
        out_batch_splits->get_underlying()->map(true);
    }
    if (out_keeps) {
      out_keeps->get_underlying()->map(true);
      out_keeps_size->get_underlying()->map(true);
    }
    // Run
    box_with_nms_limit_layer_.run();
    tscores_->get_underlying()->unmap();
    tboxes_->get_underlying()->unmap();
    if (batch_splits_) {
      batch_splits_->get_underlying()->unmap();
    }
    out_scores->get_underlying()->unmap();
    out_boxes->get_underlying()->unmap();
    out_classes->get_underlying()->unmap();
    if (out_batch_splits) {
        out_batch_splits->get_underlying()->unmap();
    }
    if (out_keeps) {
      out_keeps->get_underlying()->unmap();
      out_keeps_size->get_underlying()->unmap();
    }
  } else {
    bool scores_need_allocation = out_scores->Resize(max_total_keeps);
    bool boxes_need_allocation = out_boxes->Resize(max_total_keeps, 4);
    bool classes_need_allocation = out_classes->Resize(max_total_keeps);
    bool out_batch_splits_need_allocation = false;
    bool keeps_need_allocation            = false;
    bool keeps_size_need_allocation       = false;
    if (out_batch_splits) {
      out_batch_splits_need_allocation = out_batch_splits->Resize(batch_size);
    }
    if (out_keeps) {
      keeps_need_allocation = out_keeps->Resize(detections_per_im_);
      keeps_size_need_allocation = out_keeps_size->Resize(batch_size, num_classes);
    }
    // Configure
    box_with_nms_limit_layer_.configure(tscores_->get_underlying(), tboxes_->get_underlying(),
                                        (batch_splits_) ? batch_splits_->get_underlying() : nullptr,
                                        out_scores->get_underlying(), out_boxes->get_underlying(), out_classes->get_underlying(),
                                        (out_batch_splits) ? out_batch_splits->get_underlying() : nullptr,
                                        (out_keeps) ? out_keeps->get_underlying() : nullptr,
                                        (out_keeps_size) ? out_keeps_size->get_underlying() : nullptr,
                                        info_);
    // Allocate
    tscores_->lazy_allocate(tscoresblob, second_run_, true);
    tboxes_->lazy_allocate(tboxesblob, second_run_, true);
    if (batch_splits_) {
      batch_splits_->lazy_allocate(batchsplitsblob, second_run_, true);
    }
    if (scores_need_allocation) {
      out_scores->allocate();
    }
    if (boxes_need_allocation) {
      out_boxes->allocate();
    }
    if (classes_need_allocation) {
      out_classes->allocate();
    }
    if (out_batch_splits_need_allocation) {
      out_batch_splits->allocate();
    }
    if (keeps_need_allocation) {
      out_keeps->allocate();
    }
    if (keeps_size_need_allocation) {
      out_keeps_size->allocate();
    }
    // Since this is a CPP kernel/function we need to map and unmap tensors
    tscores_->get_underlying()->map(true);
    tboxes_->get_underlying()->map(true);
    if (batch_splits_) {
      batch_splits_->get_underlying()->map(true);
    }
    out_scores->get_underlying()->map(true);
    out_boxes->get_underlying()->map(true);
    out_classes->get_underlying()->map(true);
    if (out_batch_splits) {
      out_batch_splits->get_underlying()->map(true);
    }
    if (out_keeps) {
      out_keeps->get_underlying()->map(true);
      out_keeps_size->get_underlying()->map(true);
    }
    // Run
    box_with_nms_limit_layer_.run();
    // Unmap
    tscores_->get_underlying()->unmap();
    tboxes_->get_underlying()->unmap();
    if (batch_splits_) {
      batch_splits_->get_underlying()->unmap();
    }
    out_scores->get_underlying()->unmap();
    out_boxes->get_underlying()->unmap();
    out_classes->get_underlying()->unmap();
    if (out_batch_splits) {
      out_batch_splits->get_underlying()->unmap();
    }
    if (out_keeps) {
      out_keeps->get_underlying()->unmap();
      out_keeps_size->get_underlying()->unmap();
    }
  }

  return true;
}

REGISTER_CL_OPERATOR(BoxWithNMSLimit, CPPBoxWithNMSLimitOp<DataType>);

} // namespace caffe2
