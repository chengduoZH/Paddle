cc_library(fuse_optimizer_op_pass SRCS fuse_optimizer_op_pass.cc DEPS graph graph_helper)
cc_library(fuse_adam_op_pass SRCS fuse_adam_op_pass.cc DEPS fuse_optimizer_op_pass)
cc_library(fuse_sgd_op_pass SRCS fuse_sgd_op_pass.cc DEPS fuse_optimizer_op_pass)
cc_library(fuse_momentum_op_pass SRCS fuse_momentum_op_pass.cc DEPS fuse_optimizer_op_pass)

cc_test(fuse_optimizer_op_pass_test SRCS fuse_optimizer_op_pass_test.cc DEPS graph pass
        graph_helper fuse_sgd_op_pass enforce pass)
