include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(TEST_SOURCES
  unit/test_linear.cpp
  unit/test_activation.cpp
  unit/test_mse.cpp
  unit/test_bce.cpp
  unit/test_sgd.cpp
  unit/test_adam.cpp
  unit/test_lr_schedule.cpp
  unit/test_data_loader.cpp
  unit/test_network.cpp
  unit/test_train.cpp
)

add_executable(nn_tests ${TEST_SOURCES})
target_link_libraries(nn_tests PRIVATE neuralnetwork GTest::gtest_main)
if (MSVC)
  target_compile_options(nn_tests PRIVATE /W4)
else()
  target_compile_options(nn_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(GoogleTest)
gtest_discover_tests(nn_tests PROPERTIES LABELS "nn")
