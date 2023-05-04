cmake_minimum_required(VERSION 3.5)
project(ji)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)


# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)

# project include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# source
file(GLOB OCR_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(OCR_COMPILE_CODE ${OCR_SRC})

add_library(${PROJECT_NAME} SHARED ${OCR_COMPILE_CODE} )
target_link_libraries(${PROJECT_NAME}  
        "/usr/local/ev_sdk/3rd/onnxruntime/lib/libonnxruntime.so"
        ${OpenCV_LIBS}
        )
target_include_directories(${PROJECT_NAME}  
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include 
        PUBLIC ${OpenCV_INCLUDE_DIRS}
        PUBLIC "/usr/local/ev_sdk/3rd/onnxruntime/include"
        )   

# 把动态库放到指定路径
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so DESTINATION lib) 
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so.1.7.0 DESTINATION lib) 

# 生成可执行文件
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/include
)
add_executable(test-ji-api ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp)
target_link_libraries(test-ji-api
        ${OpenCV_LIBS}
        ${PROJECT_NAME}
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime/lib/libonnxruntime.so
        )
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(TARGETS test-ji-api DESTINATION bin)