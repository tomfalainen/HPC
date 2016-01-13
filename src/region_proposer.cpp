#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include "region_proposer.hpp"
#include <boost/python/stl_iterator.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

namespace py = boost::python;

//py::tuple RegionProposer::extract_regions(const py::object& img, const py::list& c_range, const py::list& r_range){
py::tuple RegionProposer::extract_regions(const cv::Mat& img, const py::list& c_range, const py::list& r_range){
	std::vector<int> v(py::stl_input_iterator<int>(c_range), py::stl_input_iterator<int>());
	return py::make_tuple(0, 5);
}

int RegionProposer::testhest(int i){
	return i;
}
/*
BOOST_PYTHON_MODULE(libWordDetection){
	boost::python::class_<RegionProposer>("pyWordDetection")
		.def("extract_regions", &RegionProposer::extract_regions)
		.def("testhest", &RegionProposer::testhest);
}
*/
static void init_ar(){
	Py_Initialize();

	import_array();
}


BOOST_PYTHON_MODULE(libWordDetection){
	//using namespace XM;
	init_ar();

	//initialize converters
	py::to_python_converter<cv::Mat,
	pbcvt::matToNDArrayBoostConverter>();
	pbcvt::matFromNDArrayBoostConverter();

	//expose module-level functions
	//def("dot", dot);
	//def("dot2", dot2);

	boost::python::class_<RegionProposer>("pyWordDetection")
		.def("extract_regions", &RegionProposer::extract_regions)
		.def("testhest", &RegionProposer::testhest);
}
