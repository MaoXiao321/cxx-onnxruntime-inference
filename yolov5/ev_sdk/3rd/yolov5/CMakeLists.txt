cmake_minimum_required(VERSION 3.5)
project(yolov5)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)

# 编译xxx.so
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB yolov5_src ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(MIDAS_COMPILE_CODE ${yolov5_src})

# 生成动态库libxxx.so（linux后缀名是.so，macos是.dylib）
add_library(${PROJECT_NAME} SHARED ${MIDAS_COMPILE_CODE})
# 添加库文件
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        "/usr/local/ev_sdk/3rd/onnxruntime/lib/libonnxruntime.so"
        )
# 添加头文件
target_include_directories(${PROJECT_NAME} 
    PUBLIC ${OpenCV_INCLUDE_DIR}
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include 
    PUBLIC "/usr/local/ev_sdk/3rd/onnxruntime/include"
)

# 将xxx.so插入到lib路径下
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
