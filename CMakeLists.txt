cmake_minimum_required(VERSION 2.8)

project(IslandCounter)

#set(BOOST_ROOT /sw/apps/build/boost/1.59.0_gcc4.9.2/tintin)
find_package(Boost COMPONENTS python REQUIRED)

find_package(PythonLibs REQUIRED)

#Manually set python path due to crappy FindPythonLibs
#set(PYTHON_ROOT /sw/comp/python/2.7.6_tintin)
#set(PYTHON_INCLUDE_DIR ${PYTHON_ROOT}/include/python2.7)
#set(PYTHON_LIBRARY ${PYTHON_ROOT}/lib/libpython2.7.so)

find_package( OpenCV REQUIRED )

set(BOOST_NO_BOOST_CMAKE ON)

add_executable( test main.cpp )
#add_library(IslandCounter SHARED islandCounter.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

#target_include_directories(IslandCounter PUBLIC ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(IslandCounter LINK_PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARY})
