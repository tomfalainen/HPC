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

int get_most_common_value(const Mat& img){
    double min, m;
    minMaxLoc(img, &min, &m);
    std::vector<int> counts(static_cast<int>(m) + 1, 0);

    for(int r = 0; r < img.rows; ++r){
        for(int c = 0; c < img.cols; ++c){
            int val = img.at<int>(r, c);
            if(val > 0){
                ++counts[val];
            }
        }
    }
    return (std::max_element(std::begin(counts), std::end(counts)) - std::begin(counts));
}

void equals_value(Mat& img, int val){
    for(int r = 0; r < img.rows; ++r){
        for(int c = 0; c < img.cols; ++c){
            int p = img.at<int>(r, c);
            if(p == val){
                img.at<int>(r, c) = 1;
            }else{
                img.at<int>(r, c) = 0;
            }
        }
    }    
    return;
}

cv::Mat test_equals_value(const cv::Mat& img, int val){
    Mat dst;
    img.copyTo(dst);
    equals_value(dst, val);
    return dst;
}

void _mask_p_img(Mat& img, const Mat& mask){
    for(int r = 0; r < img.rows; ++r){
        for(int c = 0; c < img.cols; ++c){
//            int p = img.at<int>(r, c);
            if(mask.at<int>(r, c) == 0){
                img.at<int>(r, c) = 0;
            }
        }
    }    
    return;
}

Mat test_mask_p_img(const Mat& img, const Mat& mask){
    Mat dst = img.clone();
    _mask_p_img(dst, mask);
    return dst;
//    return py::make_tuple(mask.type(), mask.depth(), dst.type(), dst.depth());
}

Mat calculate_overlap_fg(const Mat& boxes1, const Mat& boxes2, const Mat& img, const Mat& gt_img,
                      const Mat& params1, const Mat& params2){
    Mat b1, b2, overlaps;
    cv::Mat s_img, l_img, gt, tmp_img, p_img, i_img, element, mask;
    overlaps = Mat::zeros(boxes1.rows, boxes2.rows, CV_32F);
    for(int i = 0; i < boxes1.rows; ++i){
        b1 = boxes1.row(i);
        for(int j = 0; j < boxes2.rows; ++j){
            b2 = boxes2.row(j);

            int x1 = std::max(b1.at<int>(0), b2.at<int>(0));
            int x2 = std::min(b1.at<int>(2), b2.at<int>(2));
            int y1 = std::max(b1.at<int>(1), b2.at<int>(1));
            int y2 = std::min(b1.at<int>(3), b2.at<int>(3));
            
            //find area of intersection
            int ai = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    
            if(ai == 0){
                continue;
            }else{      //Calculate foreground overlap
//              The assumption is that the most common value is the image value for which the box was created
//                Mat gt;
                gt_img(Range(b2.at<int>(1), b2.at<int>(3)), Range(b2.at<int>(0), b2.at<int>(2))).copyTo(gt);
                int gt_val = get_most_common_value(gt);
                equals_value(gt, gt_val);

//              This is needed so that we know which connected components are supposed to be marked as the same word                
                Mat p1 = params1.row(i);
        	     element = getStructuringElement(MORPH_RECT, Size(p1.at<int>(1), p1.at<int>(0)));
                tmp_img = img(Range(b1.at<int>(1), b1.at<int>(3)), Range(b1.at<int>(0), b1.at<int>(2))).clone();

			/// Apply the specified morphology operation
			morphologyEx(tmp_img, s_img, MORPH_CLOSE, element);
		     int n = connectedComponents(s_img, l_img, 4);
//                Mat p_img;
                l_img.copyTo(p_img);

//              proposal box
//              zero out excess pixels
//                Mat mask;   
                img(Range(b1.at<int>(1), b1.at<int>(3)), Range(b1.at<int>(0), b1.at<int>(2))).convertTo(mask, gt_img.type());
                _mask_p_img(p_img, mask);
                if(sum(p_img)[0] == 0){
                    continue;
                }

                int val = get_most_common_value(p_img);
                equals_value(p_img, val);

//              intersection
                mask = gt_img(Range(y1, y2), Range(x1, x2));
                equals_value(mask, gt_val);
                i_img = l_img(Range(y1 - b1.at<int>(1), y2 - b1.at<int>(1)), 
                                  Range(x1 - b1.at<int>(0), x2 - b1.at<int>(0)));

                i_img = i_img.mul(mask); 
                equals_value(i_img, val);

                int a1 = static_cast<int>(sum(p_img)[0]);
                int a2 = static_cast<int>(sum(gt)[0]);
                ai = static_cast<int>(sum(i_img)[0]);

//              calculate union    
                int au = a1 + a2 - ai;
                
                if(au == 0){
                    continue;
                }
                    
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
    py::def("calculate_overlap_fg", calculate_overlap_fg);
    py::def("get_most_common_value", get_most_common_value);
    py::def("test_equals_value", test_equals_value);
    py::def("test_mask_p_img", test_mask_p_img);
}
