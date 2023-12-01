#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void stack_ball_query_forward_npu(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx) {
  int64_t nsample_i64 = nsample;
  at::Tensor xyz_transpose = xyz.transpose(0, 1);

  EXEC_NPU_CMD(aclnnStackBallQuery, xyz_transpose, new_xyz, xyz_batch_cnt, new_xyz_batch_cnt,idx,max_radius, nsample_i64);
}

void stack_ball_query_forward_impl(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx);

REGISTER_NPU_IMPL(stack_ball_query_forward_impl, stack_ball_query_forward_npu);
