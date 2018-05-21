cc_library(ThreadLocal SRCS ThreadLocal.cc)
cc_library(State SRCS State.cc DEPS ThreadLocal)
