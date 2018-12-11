#include "cl_operator_test.h"

namespace caffe2 {

TEST(OPENCLOperatorTest, ChannelShuffle) {
  for (auto input_dims: std::vector<std::vector<int64_t>>({
        {1, 16, 20, 20},
        {2, 16, 20, 20},
        {1, 24, 20, 20},
        {1, 128, 23, 20},
        {1, 112, 23, 20},
          })) {
    for (auto groups: std::vector<int>({2, 4, 8})){
      Workspace ws;

      PopulateCPUBlob(&ws, true, "cpu_X", input_dims);

      NetDef cpu_net;
      {
        OperatorDef* def = AddOp(&cpu_net, "ChannelShuffle", {"cpu_X"}, {"ref_Y"});
        ADD_ARG((*def), "group", i, groups);
        ADD_ARG((*def), "kernel", i, 1);
      }

      NetDef gpu_net;
      gpu_net.set_type("opencl");
      {
        OperatorDef* def = AddOp(&gpu_net, "ChannelShuffle", {"cpu_X"}, {"gpu_Y"});
        ADD_ARG((*def), "group", i, groups);
        ADD_ARG((*def), "kernel", i, 1);
        MAKE_OPENCL_OPERATOR(def);
      }
      compareNetResult(ws, cpu_net, gpu_net);
    }
  }

}

} // namespace caffe2
