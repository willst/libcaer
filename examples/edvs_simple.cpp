#include <libcaercpp/devices/edvs.hpp>
#include <csignal>
#include <atomic>

using namespace std;

static atomic_bool globalShutdown(false);

static void globalShutdownSignalHandler(int signal) {
	// Simply set the running flag to false on SIGTERM and SIGINT (CTRL+C) for global shutdown.
	if (signal == SIGTERM || signal == SIGINT) {
		globalShutdown.store(true);
	}
}

static void serialShutdownHandler(void *ptr) {
	(void)(ptr); // UNUSED.

	globalShutdown.store(true);
}

int main(void) {
	// Install signal handler for global shutdown.
#if defined(_WIN32)
	if (signal(SIGTERM, &globalShutdownSignalHandler) == SIG_ERR) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGTERM. Error: %d.", errno);
		return (EXIT_FAILURE);
	}

	if (signal(SIGINT, &globalShutdownSignalHandler) == SIG_ERR) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGINT. Error: %d.", errno);
		return (EXIT_FAILURE);
	}
#else
	struct sigaction shutdownAction;

	shutdownAction.sa_handler = &globalShutdownSignalHandler;
	shutdownAction.sa_flags = 0;
	sigemptyset(&shutdownAction.sa_mask);
	sigaddset(&shutdownAction.sa_mask, SIGTERM);
	sigaddset(&shutdownAction.sa_mask, SIGINT);

	if (sigaction(SIGTERM, &shutdownAction, NULL) == -1) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGTERM. Error: %d.", errno);
		return (EXIT_FAILURE);
	}

	if (sigaction(SIGINT, &shutdownAction, NULL) == -1) {
		libcaer::log::log(libcaer::log::logLevel::CRITICAL, "ShutdownAction",
			"Failed to set signal handler for SIGINT. Error: %d.", errno);
		return (EXIT_FAILURE);
	}
#endif

	// Open an EDVS-4337, give it a device ID of 1, on the default Linux USB serial port.
	libcaer::devices::edvs edvsHandle = libcaer::devices::edvs(1, "/dev/ttyUSB0",
		CAER_HOST_CONFIG_SERIAL_BAUD_RATE_12M);

	// Let's take a look at the information we have on the device.
	struct caer_edvs_info edvs_info = edvsHandle.infoGet();

	printf("%s --- ID: %d, Master: %d, DVS X: %d, DVS Y: %d.\n", edvs_info.deviceString, edvs_info.deviceID,
		edvs_info.deviceIsMaster, edvs_info.dvsSizeX, edvs_info.dvsSizeY);

	// Send the default configuration before using the device.
	// No configuration is sent automatically!
	edvsHandle.sendDefaultConfig();

	// Tweak some biases, to increase bandwidth in this case.
	edvsHandle.configSet(EDVS_CONFIG_BIAS, EDVS_CONFIG_BIAS_PR, 695);
	edvsHandle.configSet(EDVS_CONFIG_BIAS, EDVS_CONFIG_BIAS_FOLL, 867);

	// Let's verify they really changed!
	uint32_t prBias = edvsHandle.configGet(EDVS_CONFIG_BIAS, EDVS_CONFIG_BIAS_PR);
	uint32_t follBias = edvsHandle.configGet(EDVS_CONFIG_BIAS, EDVS_CONFIG_BIAS_FOLL);

	printf("New bias values --- PR: %d, FOLL: %d.\n", prBias, follBias);

	// Now let's get start getting some data from the device. We just loop in blocking mode,
	// no notification needed regarding new events. The shutdown notification, for example if
	// the device is disconnected, should be listened to.
	edvsHandle.dataStart(nullptr, nullptr, nullptr, &serialShutdownHandler, nullptr);

	// Let's turn on blocking data-get mode to avoid wasting resources.
	edvsHandle.configSet(CAER_HOST_CONFIG_DATAEXCHANGE, CAER_HOST_CONFIG_DATAEXCHANGE_BLOCKING, true);

	while (!globalShutdown.load(memory_order_relaxed)) {
		std::unique_ptr<libcaer::events::EventPacketContainer> packetContainer = edvsHandle.dataGet();
		if (packetContainer == nullptr) {
			continue; // Skip if nothing there.
		}

		printf("\nGot event container with %d packets (allocated).\n", packetContainer->size());

		for (auto &packet : *packetContainer) {
			if (packet == nullptr) {
				printf("Packet is empty (not present).\n");
				continue; // Skip if nothing there.
			}

			printf("Packet of type %d -> %d events, %d capacity.\n", packet->getEventType(), packet->getEventNumber(),
				packet->getEventCapacity());

			if (packet->getEventType() == POLARITY_EVENT) {
				std::shared_ptr<const libcaer::events::PolarityEventPacket> polarity = std::static_pointer_cast<
					libcaer::events::PolarityEventPacket>(packet);

				// Get full timestamp and addresses of first event.
				const libcaer::events::PolarityEvent &firstEvent = (*polarity)[0];

				int32_t ts = firstEvent.getTimestamp();
				uint16_t x = firstEvent.getX();
				uint16_t y = firstEvent.getY();
				bool pol = firstEvent.getPolarity();

				printf("First polarity event - ts: %d, x: %d, y: %d, pol: %d.\n", ts, x, y, pol);
			}
		}
	}

	edvsHandle.dataStop();

	// Close automatically done by destructor.

	printf("Shutdown successful.\n");

	return (EXIT_SUCCESS);
}
