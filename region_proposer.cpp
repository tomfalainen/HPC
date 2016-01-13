#include "region_proposer.hpp"
#include <boost/python/stl_iterator.hpp>

namespace py = boost::python;

boost::python::tuple RegionProposer::extract_regions(const boost::python::object& img, const boost::python::list& c_range, const boost::python::list& r_range){
	std::vector<int> v(py::stl_input_iterator<int>(c_range), py::stl_input_iterator<int>());
	return py::make_tuple(0, 5);
}

int RegionProposer::testhest(int i){
	return i;
}

BOOST_PYTHON_MODULE(libWordDetection){
	boost::python::class_<RegionProposer>("pyWordDetection")
		.def("extract_regions", &RegionProposer::extract_regions)
		.def("testhest", &RegionProposer::testhest);
}

