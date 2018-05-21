cc_library(thread_local SRCS ThreadLocal.cc)
cc_library(state SRCS State.cc DEPS thread_local)
