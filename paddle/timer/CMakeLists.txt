cc_library(thread_local SRCS thread_local.cc)
cc_library(state SRCS state.cc DEPS thread_local)
