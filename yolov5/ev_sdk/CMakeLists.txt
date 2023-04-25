cmake_minimum_required(VERSION 3.5)
project(ji)
set(CMAKE_CXX_STANDARD 11)

#以下指定动态库的查找地址
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)


# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)

# 把动态库放到指定路径
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so DESTINATION lib) 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so.1.7.0 DESTINATION lib) 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/yolov5/lib/libyolov5.so DESTINATION lib) 

# 生成可执行文件
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/yolov5/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/include
)
add_executable(test-ji-api ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)
target_link_libraries(test-ji-api
        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/yolov5/lib/libyolov5.so
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so
        )
install(TARGETS test-ji-api DESTINATION bin)