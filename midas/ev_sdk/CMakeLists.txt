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
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so.1.6.0 DESTINATION lib) 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/midas/lib/libmidas.so DESTINATION lib) 


# 生成可执行文件
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/midas/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/include
)
add_executable(test-midas-api ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)
target_link_libraries(test-midas-api
        ${OpenCV_LIBS}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/midas/lib/libmidas.so
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so
        )
install(TARGETS test-midas-api DESTINATION bin)