#ifndef LIBCAER_EVENTS_UTILS_HPP_
#define LIBCAER_EVENTS_UTILS_HPP_

#include "common.hpp"
#include "config.hpp"
#include "ear.hpp"
#include "frame.hpp"
#include "imu6.hpp"
#include "imu9.hpp"
#include "point1d.hpp"
#include "point2d.hpp"
#include "point3d.hpp"
#include "point4d.hpp"
#include "polarity.hpp"
#include "sample.hpp"
#include "special.hpp"
#include "spike.hpp"
#include <memory>

namespace libcaer {
namespace events {
namespace utils {

std::unique_ptr<EventPacket> makeUniqueFromCStruct(caerEventPacketHeader packet);
std::shared_ptr<EventPacket> makeSharedFromCStruct(caerEventPacketHeader packet);

std::unique_ptr<EventPacket> makeUniqueFromCStruct(caerEventPacketHeader packet) {
	switch (caerEventPacketHeaderGetEventType(packet)) {
		case SPECIAL_EVENT:
			return (std::unique_ptr<SpecialEventPacket>(new SpecialEventPacket(packet)));
			break;

		case POLARITY_EVENT:
			return (std::unique_ptr<PolarityEventPacket>(new PolarityEventPacket(packet)));
			break;

		case FRAME_EVENT:
			return (std::unique_ptr<FrameEventPacket>(new FrameEventPacket(packet)));
			break;

		case IMU6_EVENT:
			return (std::unique_ptr<IMU6EventPacket>(new IMU6EventPacket(packet)));
			break;

		case IMU9_EVENT:
			return (std::unique_ptr<IMU9EventPacket>(new IMU9EventPacket(packet)));
			break;

		case SAMPLE_EVENT:
			return (std::unique_ptr<SampleEventPacket>(new SampleEventPacket(packet)));
			break;

		case EAR_EVENT:
			return (std::unique_ptr<EarEventPacket>(new EarEventPacket(packet)));
			break;

		case CONFIG_EVENT:
			return (std::unique_ptr<ConfigurationEventPacket>(new ConfigurationEventPacket(packet)));
			break;

		case POINT1D_EVENT:
			return (std::unique_ptr<Point1DEventPacket>(new Point1DEventPacket(packet)));
			break;

		case POINT2D_EVENT:
			return (std::unique_ptr<Point2DEventPacket>(new Point2DEventPacket(packet)));
			break;

		case POINT3D_EVENT:
			return (std::unique_ptr<Point3DEventPacket>(new Point3DEventPacket(packet)));
			break;

		case POINT4D_EVENT:
			return (std::unique_ptr<Point4DEventPacket>(new Point4DEventPacket(packet)));
			break;

		case SPIKE_EVENT:
			return (std::unique_ptr<SpikeEventPacket>(new SpikeEventPacket(packet)));
			break;

		default:
			return (std::unique_ptr<EventPacket>(new EventPacket(packet)));
			break;
	}
}

std::shared_ptr<EventPacket> makeSharedFromCStruct(caerEventPacketHeader packet) {
	switch (caerEventPacketHeaderGetEventType(packet)) {
		case SPECIAL_EVENT:
			return (std::make_shared<SpecialEventPacket>(packet));
			break;

		case POLARITY_EVENT:
			return (std::make_shared<PolarityEventPacket>(packet));
			break;

		case FRAME_EVENT:
			return (std::make_shared<FrameEventPacket>(packet));
			break;

		case IMU6_EVENT:
			return (std::make_shared<IMU6EventPacket>(packet));
			break;

		case IMU9_EVENT:
			return (std::make_shared<IMU9EventPacket>(packet));
			break;

		case SAMPLE_EVENT:
			return (std::make_shared<SampleEventPacket>(packet));
			break;

		case EAR_EVENT:
			return (std::make_shared<EarEventPacket>(packet));
			break;

		case CONFIG_EVENT:
			return (std::make_shared<ConfigurationEventPacket>(packet));
			break;

		case POINT1D_EVENT:
			return (std::make_shared<Point1DEventPacket>(packet));
			break;

		case POINT2D_EVENT:
			return (std::make_shared<Point2DEventPacket>(packet));
			break;

		case POINT3D_EVENT:
			return (std::make_shared<Point3DEventPacket>(packet));
			break;

		case POINT4D_EVENT:
			return (std::make_shared<Point4DEventPacket>(packet));
			break;

		case SPIKE_EVENT:
			return (std::make_shared<SpikeEventPacket>(packet));
			break;

		default:
			return (std::make_shared<EventPacket>(packet));
			break;
	}
}

}
}
}

#endif /* LIBCAER_EVENTS_UTILS_HPP_ */
