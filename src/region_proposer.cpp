#define PY_ARRAY_UNIQUE_SYMBOL pbcvt_ARRAY_API
#include <algorithm>

#include "region_proposer.hpp"
#include <boost/python/stl_iterator.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/detail/api_placeholder.hpp>
#include <pyboostcvconverter/pyboostcvconverter.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>

namespace py = boost::python;
using namespace cv;

py::tuple extract_regions(const cv::Mat& img, const py::list& c_range, const py::list& r_range){
	py::list params, all_boxes;
	int rl = py::len(r_range);
	int cl = py::len(r_range);
	cv::Mat s_img, l_img, stats, centroids;
	for(int i = 0; i < rl; ++i){
		int r = py::extract<int>(r_range[i]);
		for(int j = 0; j < cl; ++j){
			int c = py::extract<int>(c_range[j]);		
			cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(c, r));

			/// Apply the specified morphology operation
			morphologyEx(img, s_img, cv::MORPH_CLOSE, element);
			int n = cv::connectedComponentsWithStats(s_img, l_img, stats, centroids, 4);

			for(int i = 0; i < n; ++i){
				params.append(py::make_tuple(r, c));
				auto tmp = stats.row(i);
				all_boxes.append(py::make_tuple(tmp.at<int>(0), tmp.at<int>(1), tmp.at<int>(0) + tmp.at<int>(2), tmp.at<int>(1) + tmp.at<int>(3)));
			}
		}
	}

	return py::make_tuple(all_boxes, params);
}

Mat calculate_overlap(const Mat& boxes1, const Mat& boxes2){
    Mat b1, b2, overlaps;
    overlaps = Mat::zeros(boxes1.rows, boxes2.rows, CV_32F);
    for(int i = 0; i < boxes1.rows; ++i){
        b1 = boxes1.row(i);
        for(int j = 0; j < boxes2.rows; ++j){
            b2 = boxes2.row(j);

            //calculate box area
            int a1 = (b1.at<int>(2) - b1.at<int>(0)) * (b1.at<int>(3) - b1.at<int>(1));
            int a2 = (b2.at<int>(2) - b2.at<int>(0)) * (b2.at<int>(3) - b2.at<int>(1));
    
            //find area of intersection
            int ai = std::max(0, std::min(b1.at<int>(2), b2.at<int>(2)) - std::max(b1.at<int>(0), b2.at<int>(0))) * 
                     std::max(0, std::min(b1.at<int>(3), b2.at<int>(3)) - std::max(b1.at<int>(1), b2.at<int>(1)));
    
            int au = a1 + a2 - ai;
            if(au == 0){
                overlaps.at<float>(i, j) = 0.0;
            }else{
                overlaps.at<float>(i, j) = static_cast<float>(ai) / static_cast<float>(au);
            }
        }
    }

    return overlaps;
}



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
	
    def("extract_regions", extract_regions);
    py::def("calculate_overlap", calculate_overlap);
}
