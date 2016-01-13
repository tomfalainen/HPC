cmake_minimum_required(VERSION 3.0)

project(WordDetection)

find_package(Boost COMPONENTS python REQUIRED)

find_package(PythonLibs REQUIRED)

find_package( OpenCV REQUIRED )

#set(BOOST_NO_BOOST_CMAKE ON)

file(GLOB project_sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable( region_proposer_test src/main.cpp )
add_library(WordDetection SHARED ${project_sources})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_include_directories(WordDetection PUBLIC ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/include") 
target_link_libraries(WordDetection LINK_PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARY})


#target_include_directories(test PUBLIC ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(region_proposer_test LINK_PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARY})
