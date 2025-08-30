target_sources( layer_norm PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/device_tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/host_tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)