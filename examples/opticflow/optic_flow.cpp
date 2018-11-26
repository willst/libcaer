#include <iostream>

#include "Eigen/Dense"

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/eigen.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

using namespace cv;
using namespace std;

//command line to buld shared library
//g++ -std=c++11 -fPIC -shared -Wno-undef -I/usr/include/python2.7 -I/usr/include/eigen3 -I/usr/local/include/pybind11 -I/usr/lib/python2.7/site-packages/numpy/core/include -O3 $(pkg-config --cflags-only-I opencv) $(pkg-config --libs opencv) optic_flow.cpp -o optic_flow.so

template <class T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> convertCV (const cv::Mat_<T>& src) {
    if ( src.empty() )
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>();

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dst(src.rows, src.cols);

    cv2eigen(src, dst);
    return dst;
}


static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
                    double, const Scalar& line_color, double, const Scalar& circle_color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 line_color);
            circle(cflowmap, Point(x,y), 2, circle_color, -1);
        }
}

static void drawOptFlowMap(vector<Point2f>& features_p, vector<Point2f>& features_n, Mat& cflowmap,
                    double, const Scalar& line_color, double, const Scalar& circle_color)
{
    vector<Point2f>::iterator v_p = features_p.begin();
    vector<Point2f>::iterator v_n = features_n.begin();

    for(; v_p != features_p.end() && v_n != features_n.end(); ++v_p, ++v_n)
        {
            Point2f p = Point(cvRound(v_p->x),cvRound(v_p->y));
            Point2f n = Point(cvRound(v_n->x), cvRound(v_n->y));
            if (p != n)
                line(cflowmap, p, n, line_color);
            else
                circle(cflowmap, p, 2, circle_color, -1);
        }
}

static void calcOptFlowMap(const Mat& flow,
                     Mat1f& flow_vx, Mat1f& flow_vy)
{
    flow_vx = Mat1f::zeros(flow_vx.rows, flow_vx.cols);
    flow_vy = Mat1f::zeros(flow_vy.rows, flow_vy.cols);

    for(int y = 0; y < flow.rows; y ++)
        for(int x = 0; x < flow.cols; x ++)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);

            flow_vx[y][x] = fxy.x;
            flow_vy[y][x] = fxy.y;
        }
}

static void calcOptFlowMap(vector<Point2f>& features_p, vector<Point2f>& features_n,
                     Mat1f& flow_vx, Mat1f& flow_vy, Mat1i& flow_count)
{
    flow_vx = Mat1f::zeros(flow_vx.rows, flow_vx.cols);
    flow_vy = Mat1f::zeros(flow_vy.rows, flow_vy.cols);
    flow_count = Mat1i::zeros(flow_count.rows, flow_count.cols);

    vector<Point2f>::iterator v_p = features_p.begin();
    vector<Point2f>::iterator v_n = features_n.begin();

    for(; v_p != features_p.end() && v_n != features_n.end(); ++v_p, ++v_n)
        {
            const int x = cvRound(v_p->x);
            const int y = cvRound(v_p->y);

            flow_vx[y][x] += v_n->x - v_p -> x;
            flow_vy[y][x] += v_n->y - v_p -> y;
            flow_count[y][x] ++;
        }
}

static const Scalar LINE_COLOR = Scalar(0, 255, 0);
static const Scalar CIRCLE_COLOR = Scalar(255, 0, 0);

class OpticFlow
{
    private:
        Mat flow, cflow, frame;
        UMat gray, prevgray, uflow;
        vector<cv::Point2f> features_prev, features_next;

        Mat1f flow_vx, flow_vy;
        Mat1i flow_count;

        VideoCapture cap;
        bool do_use_sparse;
        bool do_draw_vectors;

    public:
        OpticFlow(const bool use_sparse = true, const bool draw_vectors = true) {

            do_use_sparse = use_sparse;
            do_draw_vectors = draw_vectors;

            cap = VideoCapture("visiontraffic.avi");

            if(!cap.isOpened())
                exit (-1);

            cap >> frame;

            flow_vx = Mat1f(frame.rows,frame.cols);
            flow_vy = Mat1f(frame.rows,frame.cols);
            flow_count = Mat1i(frame.rows,frame.cols);

            if (do_draw_vectors)
                namedWindow("flow", 1);

        };

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getVx() { return convertCV(flow_vx); }
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getVy() { return convertCV(flow_vy); }
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getCount() { return convertCV(flow_count); }

        bool processFrame() {
            cap >> frame;
            if ( frame.empty() ) {
                std::cout << "Video frame empty; likely end of file. \n";
                return false;
            }

            if (do_use_sparse)
                return processFrameSparse(frame);
            else
                return processFrameDense(frame);
        }

        bool processFrameDense(const Mat& frame) {

            cvtColor(frame, gray, COLOR_BGR2GRAY);

            if( !prevgray.empty() )
            {
                calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
                cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
                uflow.copyTo(flow);
                calcOptFlowMap(flow, flow_vx, flow_vy);

                if (do_draw_vectors) {
                    drawOptFlowMap(flow, cflow, 16, 1.5, LINE_COLOR, 1.5, CIRCLE_COLOR);
                    imshow("flow", cflow);
                }
            }
            if(waitKey(1)>=0)
                return false;

            cv::swap(prevgray, gray);
            return true;
        };

        bool processFrameSparse(const Mat& frame) {
            vector<uchar> status;
            vector<float> err;

            cvtColor(frame, gray, COLOR_BGR2GRAY);

            if( !prevgray.empty() )
            {
                cv::goodFeaturesToTrack(prevgray, // the image
                  features_prev,   // the output detected features
                  1000,  // the maximum number of features
                  0.01,     // quality level
                  10     // min distance between two features
                );


                if ( !features_prev.empty() ) {
                    calcOpticalFlowPyrLK(prevgray, gray, features_prev, features_next, status, err);
                    cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
                    calcOptFlowMap(features_prev, features_next, flow_vx, flow_vy, flow_count);

                    if (do_draw_vectors) {
                        drawOptFlowMap(features_prev, features_next, cflow, 1.5, LINE_COLOR, 1.5, CIRCLE_COLOR);
                        imshow("flow", cflow);
                    }
                }
            }
            if(waitKey(1)>=0)
                return false;

            std::swap(features_prev, features_next);
            cv::swap(prevgray, gray);
            return true;
        };



};


PYBIND11_MODULE(optic_flow, m) {

    py::class_<OpticFlow>(m, "OpticFlow")
        .def(py::init<>())
        .def("getVx", &OpticFlow::getVx, py::return_value_policy::reference_internal)
        .def("getVy", &OpticFlow::getVy, py::return_value_policy::reference_internal)
        .def("getCount", &OpticFlow::getCount, py::return_value_policy::reference_internal)
        .def("processFrame", &OpticFlow::processFrame)
    ;
};


int main()
{
    OpticFlow optic_flow = OpticFlow();

    for(;;)
    {
        if (!optic_flow.processFrame())
            break;
    }
    return 0;
}