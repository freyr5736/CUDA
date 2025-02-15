cmake_minimum_required(VERSION 3.18)
project(image_processor LANGUAGES CXX CUDA)

# Set OpenCV directory explicitly
set(OpenCV_DIR /home/freyr5736/opencv/build)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories (modify paths if necessary)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/include/opencv4
    /home/freyr5736/opencv/modules/core/include
    /home/freyr5736/opencv/modules/photo/include
)

# Add executable
add_executable(image_processor image_processor.cpp)

# Link libraries
target_link_libraries(image_processor 
    ${OpenCV_LIBS} 
)
