include(ExternalProject)

set(DLPANKSOURCE_DIR ${THIRD_PARTY_PATH}/dlpank)
set(DLPANKINCLUDE_DIR ${DLPANKSOURCE_DIR}/src/extern_dlpank)

include_directories(${DLPANKINCLUDE_DIR})

ExternalProject_Add(
  extern_dlpank
  ${EXTERNAL_PROJECT_LOG_ARGS}
  GIT_REPOSITORY "https://github.com/dmlc/dlpack.git"
  GIT_TAG        "v0.2"
  PREFIX         ${DLPANKSOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

if(${CMAKE_VERSION} VERSION_LESS "3.3.0")
  set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/dlpank_dummy.c)
  file(WRITE ${dummyfile} "const char *dummy = \"${dummyfile}\";")
  add_library(dlpank STATIC ${dummyfile})
else()
  add_library(dlpank INTERFACE)
endif()

add_dependencies(dlpank extern_dlpank)

LIST(APPEND externl_project_dependencies dlpank)
