cmake_minimum_required(VERSION 3.0)

project(IslandCounter)

#set(BOOST_ROOT /sw/apps/build/boost/1.59.0_gcc4.9.2/tintin)
find_package(Boost COMPONENTS python REQUIRED)

find_package(PythonLibs REQUIRED)

#Manually set python path due to crappy FindPythonLibs
#set(PYTHON_ROOT /sw/comp/python/2.7.6_tintin)
#set(PYTHON_INCLUDE_DIR ${PYTHON_ROOT}/include/python2.7)
#set(PYTHON_LIBRARY ${PYTHON_ROOT}/lib/libpython2.7.so)

set(PYBOOSTCONVERTER_ROOT "/home/tomas/libs/pyboostcvconverter")

find_package( OpenCV REQUIRED )

add_library(pbcvt SHARED IMPORTED)
set_target_properties(pbcvt PROPERTIES IMPORTED_LOCATION ${PYBOOSTCONVERTER_ROOT}/build)

#set(BOOST_NO_BOOST_CMAKE ON)

add_executable( region_proposer_test main.cpp )
add_library(WordDetection SHARED region_proposer.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

target_include_directories(WordDetection PUBLIC ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} "${PYBOOSTCONVERTER_ROOT}/include")
target_link_libraries(WordDetection LINK_PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARY} ${PYBOOSTCONVERTER_ROOT}/build/pbcvt.so)

#message(${PYBOOSTCONVERTER_ROOT})

#target_include_directories(test PUBLIC ${BOOST_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(region_proposer_test LINK_PUBLIC ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARY})
