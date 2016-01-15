#include <vector>

#include <boost/python.hpp>
#include <opencv2/opencv.hpp>


class RegionProposer{
	public:
//		boost::python::tuple extract_regions(const boost::python::object&, const boost::python::list&, const boost::python::list&);
		boost::python::tuple extract_regions(const cv::Mat&, const boost::python::list&, const boost::python::list&);
		int testhest(int);
};

boost::python::tuple extract_regions(const cv::Mat&, const boost::python::list&, const boost::python::list&);

cv::Mat calculate_overlap(const cv::Mat&, const cv::Mat&);
