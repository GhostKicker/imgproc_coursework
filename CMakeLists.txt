cmake_minimum_required(VERSION 3.0)

project(Fadeev_CourseWork)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(MarkingDown main_md.cpp)
target_link_libraries(MarkingDown ${OpenCV_LIBS})

add_executable(Application main_app.cpp)
target_link_libraries(Application ${OpenCV_LIBS})
