
# nv_library(pten SRCS tensor_compiler.cc DEPS  tensor tc_core tc_autotuner tc_aten  Halide tc_lang tc_proto pthread tc_core_cpu tc_cuda mkl_core)
nv_library(pten SRCS tensor_compiler.cc DEPS  tensor )
