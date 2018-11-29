#include <atomic>
#include <csignal>

#include "libcaercpp/devices/dvs128.hpp"
#include "libcaercpp/filters/dvs_noise.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

namespace py = pybind11;

#include "optic_flow.cpp"

#if !defined(LIBCAER_HAVE_OPENCV) || LIBCAER_HAVE_OPENCV == 0
#error "This example requires OpenCV support in libcaer to be enabled."
#endif

//g++ -std=c++11 -fPIC -shared -Wno-undef -O3 -I/usr/local/include/pybind11 -I/usr/lib/python2.7/site-packages/numpy/core/include $(pkg-config --cflags --libs python2 eigen3 opencv) dvs128_py.cpp -o optic_flow.so


//RasPi: export PKG_CONFIG_PATH=${CONDA_ENV_PATH}/lib/pkgconfig
//g++ -std=c++11 -fPIC -shared -Wno-undef -O3 -I/usr/local/include/pybind11 -I${CONDA_ENV_PATH}/lib/python2.7/site-packages/numpy/core/include $(pkg-config --cflags --libs python2 eigen3 opencv) dvs128_py.cpp -o optic_flow.so


using namespace std;
using namespace cv;

static atomic_bool globalShutdown(false);

static void globalShutdownSignalHandler(int signal) {
	// Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
	if (signal == SIGTERM || signal == SIGINT) {
		globalShutdown.store(true);
	}
}

static void usbShutdownHandler(void *ptr) {
	(void) (ptr); // UNUSED.

	globalShutdown.store(true);
}

namespace opticflow {

class DvsConnector
{
    private:
        libcaer::devices::dvs128 dvs128Handle = libcaer::devices::dvs128(1, 0, 0, "");
        libcaer::filters::DVSNoise dvsNoiseFilter = libcaer::filters::DVSNoise(128, 128);
        cv::Mat cvEvents = cv::Mat(128,128,CV_8UC3, cv::Vec3b{127, 127, 127});

    public:
        DvsConnector()
        {
            printf("Opening connection for DVS. \n");

            // Open a DVS128, give it a device ID of 1, and don't care about USB bus or SN restrictions.
            dvs128Handle = libcaer::devices::dvs128(1, 0, 0, "");

            // Send the default configuration before using the device.
            // No configuration is sent automatically!
            dvs128Handle.sendDefaultConfig();

            // Tweak some biases, to increase bandwidth in this case.
            dvs128Handle.configSet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PR, 695);
            dvs128Handle.configSet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_FOLL, 867);

            // Let's verify they really changed!
            uint32_t prBias   = dvs128Handle.configGet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_PR);
            uint32_t follBias = dvs128Handle.configGet(DVS128_CONFIG_BIAS, DVS128_CONFIG_BIAS_FOLL);

            printf("New bias values --- PR: %d, FOLL: %d.\n", prBias, follBias);

            // Now let's get start getting some data from the device. We just loop in blocking mode,
            // no notification needed regarding new events. The shutdown notification, for example if
            // the device is disconnected, should be listened to.
            dvs128Handle.dataStart(nullptr, nullptr, nullptr, &usbShutdownHandler, nullptr);

            // Let's turn on blocking data-get mode to avoid wasting resources.
            dvs128Handle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

            struct caer_dvs128_info dvs128_info = dvs128Handle.infoGet();

            //dvsNoiseFilter = libcaer::filters::DVSNoise(dvs128_info.dvsSizeX, dvs128_info.dvsSizeX);

            dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TWO_LEVELS, true);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_CHECK_POLARITY, true);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MIN, 2);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_SUPPORT_MAX, 8);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_TIME, 2000);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_BACKGROUND_ACTIVITY_ENABLE, true);

            dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_TIME, 200);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_REFRACTORY_PERIOD_ENABLE, true);

            dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_ENABLE, true);
            dvsNoiseFilter.configSet(CAER_FILTER_DVS_HOTPIXEL_LEARN, true);

//            cv::namedWindow("PLOT_EVENTS",
//		        cv::WindowFlags::WINDOW_AUTOSIZE | cv::WindowFlags::WINDOW_KEEPRATIO | cv::WindowFlags::WINDOW_GUI_EXPANDED);
        };

        ~DvsConnector() {
            dvs128Handle.dataStop();

//            cv::destroyWindow("PLOT_EVENTS");
	        printf("Shutdown successful.\n");
        };

        struct caer_dvs128_info getInfo() {
            return dvs128Handle.infoGet();
        };

        cv::Mat getEvents() {
            return cvEvents;
        }

        bool processPacket() {

            if (!globalShutdown.load(memory_order_relaxed)) {
                std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = dvs128Handle.dataGet();

                struct caer_dvs128_info dvs128_info = dvs128Handle.infoGet();

                if (packetContainer == nullptr) {
                    return false; // Skip if nothing there.
                }

                printf("\nGot event container with %d packets (allocated).\n", packetContainer->size());

                for (auto &packet : *packetContainer) {
                    if (packet == nullptr) {
                        printf("Packet is empty (not present).\n");
                        return false; // Skip if nothing there.
                    }

                    printf("Packet of type %d -> %d events, %d capacity.\n", packet->getEventType(), packet->getEventNumber(),
                        packet->getEventCapacity());

                    if (packet->getEventType() == POLARITY_EVENT) {
                        std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity
                            = std::static_pointer_cast<libcaer::events::PolarityEventPacket>(packet);

                        dvsNoiseFilter.apply(*polarity);

                        printf("Got polarity packet with %d events, after filtering remaining %d events.\n",
                            polarity->getEventNumber(), polarity->getEventValid());

                        // Get full timestamp and addresses of first event.
                        const libcaer::events::PolarityEvent &firstEvent = (*polarity)[0];

                        int32_t ts = firstEvent.getTimestamp();
                        uint16_t x = firstEvent.getX();
                        uint16_t y = firstEvent.getY();
                        bool pol   = firstEvent.getPolarity();

                        printf("First polarity event - ts: %d, x: %d, y: %d, pol: %d.\n", ts, x, y, pol);

                        cvEvents = cv::Mat(dvs128_info.dvsSizeY, dvs128_info.dvsSizeX, CV_8UC3, cv::Vec3b{127, 127, 127});

                        for (const auto &e : *polarity) {
                            // Discard invalid events (filtered out).
                            if (!e.isValid()) {
                                return false;
                            }

                            cvEvents.at<cv::Vec3b>(e.getY(), e.getX())
                                = e.getPolarity() ? cv::Vec3b{255, 255, 255} : cv::Vec3b{0, 0, 0};
                        }

                        return true;
                    }
                }
            }

            return false; //unexpected to reach here
        }
};

class DVSOpticFlow {
    private:
        opticflow::OpticFlow demoFlow = opticflow::OpticFlow(true, true);
        DvsConnector dvsConnector;

    public:
        DVSOpticFlow() {

        demoFlow.initFlows(128,128);

        };

        bool processFrame() {
            if (dvsConnector.processPacket()) {
                return demoFlow.processFrame(dvsConnector.getEvents());
            }
            else {
                std::cout << "Video frame empty; likely end of file. \n";
                return false;
            }

        }

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getVx() { return demoFlow.getVx(); }
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getVy() { return demoFlow.getVy(); }
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> getCount() { return demoFlow.getCount(); }


};


}; //namespace: opticflow


PYBIND11_MODULE(dvs_flow, m) {

    py::class_<opticflow::DemoOpticFlow>(m, "OpticFlow")
        .def(py::init<const std::string &>())
        .def("getVx", &opticflow::DemoOpticFlow::getVx, py::return_value_policy::reference_internal)
        .def("getVy", &opticflow::DemoOpticFlow::getVy, py::return_value_policy::reference_internal)
        .def("getCount", &opticflow::DemoOpticFlow::getCount, py::return_value_policy::reference_internal)
        .def("processFrame", &opticflow::DemoOpticFlow::processFrame)
    ;

    py::class_<opticflow::DVSOpticFlow>(m, "DVSOpticFlow")
        .def(py::init<>())
        .def("getVx", &opticflow::DVSOpticFlow::getVx, py::return_value_policy::reference_internal)
        .def("getVy", &opticflow::DVSOpticFlow::getVy, py::return_value_policy::reference_internal)
        .def("getCount", &opticflow::DVSOpticFlow::getCount, py::return_value_policy::reference_internal)
        .def("processFrame", &opticflow::DVSOpticFlow::processFrame)
    ;
};