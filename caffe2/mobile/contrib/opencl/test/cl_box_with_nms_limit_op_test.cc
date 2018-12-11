#include "cl_operator_test.h"

namespace caffe2 {

TEST(OPENCLOperatorTest, BoxWithNMSLimit) {
  Workspace ws;

  const int count       = 10;
  const int num_classes = 4;
  std::vector<int64_t> scores_dims  = {count, num_classes};
  std::vector<int64_t> classes_dims = {count, num_classes * 4};

  PopulateCPUBlob(&ws, true, "cpu_scores", scores_dims);
  PopulateCPUBlob(&ws, true, "cpu_classes", classes_dims);

  NetDef cpu_net;
  {
      OperatorDef* def = AddOp(&cpu_net, "BoxWithNMSLimit", {"cpu_scores", "cpu_classes"}, {"ref_scores", "ref_boxes", "ref_classes"});
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {
      OperatorDef* def = AddOp(&gpu_net, "BoxWithNMSLimit", {"cpu_scores", "cpu_classes"}, {"gpu_scores", "gpu_boxes", "gpu_classes"});
      MAKE_OPENCL_OPERATOR(def);
  }

  compareNetResult(ws, cpu_net, gpu_net, "ref_scores", "gpu_scores", 0.1);
  compareNetResult(ws, cpu_net, gpu_net, "ref_boxes", "gpu_boxes", 0.1);
  compareNetResult(ws, cpu_net, gpu_net, "ref_classes", "gpu_classes", 0.1);
}

TEST(OPENCLOperatorTest, BoxWithNMSLimit_SoftNMS) {
  Workspace ws;

  const int count       = 10;
  const int num_classes = 4;
  std::vector<int64_t> scores_dims  = {count, num_classes};
  std::vector<int64_t> classes_dims = {count, num_classes * 4};

  PopulateCPUBlob(&ws, true, "cpu_scores", scores_dims);
  PopulateCPUBlob(&ws, true, "cpu_classes", classes_dims);

  NetDef cpu_net;
  {
      OperatorDef* def = AddOp(&cpu_net, "BoxWithNMSLimit", {"cpu_scores", "cpu_classes"}, {"ref_scores", "ref_boxes", "ref_classes"});
      ADD_ARG((*def), "soft_nms_enabled", i, 1);
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {
      OperatorDef* def = AddOp(&gpu_net, "BoxWithNMSLimit", {"cpu_scores", "cpu_classes"}, {"gpu_scores", "gpu_boxes", "gpu_classes"});
      MAKE_OPENCL_OPERATOR(def);
      ADD_ARG((*def), "soft_nms_enabled", i, 1);
  }

  compareNetResult(ws, cpu_net, gpu_net, "ref_scores", "gpu_scores", 0.1);
  compareNetResult(ws, cpu_net, gpu_net, "ref_boxes", "gpu_boxes", 0.1);
  compareNetResult(ws, cpu_net, gpu_net, "ref_classes", "gpu_classes", 0.1);
}

TEST(OPENCLOperatorTest, BoxWithNMSLimit_LowDetections) {
  Workspace ws;

  const int count       = 10;
  const int num_classes = 4;
  std::vector<int64_t> scores_dims  = {count, num_classes};
  std::vector<int64_t> classes_dims = {count, num_classes * 4};

  PopulateCPUBlob(&ws, true, "cpu_scores", scores_dims);
  PopulateCPUBlob(&ws, true, "cpu_classes", classes_dims);

  NetDef cpu_net;
  {
      OperatorDef* def = AddOp(&cpu_net, "BoxWithNMSLimit", {"cpu_scores", "cpu_classes"}, {"ref_scores", "ref_boxes", "ref_classes"});
      ADD_ARG((*def), "detections_per_im", i, 5);
  }

  NetDef gpu_net;
  gpu_net.set_type("opencl");
  {
      OperatorDef* def = AddOp(&gpu_net, "BoxWithNMSLimit", {"cpu_scores", "cpu_classes"}, {"gpu_scores", "gpu_boxes", "gpu_classes"});
      MAKE_OPENCL_OPERATOR(def);
      ADD_ARG((*def), "detections_per_im", i, 5);
  }

  compareNetResult(ws, cpu_net, gpu_net, "ref_scores", "gpu_scores", 0.1);
  compareNetResult(ws, cpu_net, gpu_net, "ref_boxes", "gpu_boxes", 0.1);
  compareNetResult(ws, cpu_net, gpu_net, "ref_classes", "gpu_classes", 0.1);
}

} // namespace caffe2
