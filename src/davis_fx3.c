#include "davis_fx3.h"

static void allocateDebugTransfers(davisFX3Handle handle);
static void cancelAndDeallocateDebugTransfers(davisFX3Handle handle);
static void LIBUSB_CALL libUsbDebugCallback(struct libusb_transfer *transfer);
static void debugTranslator(davisFX3Handle handle, uint8_t *buffer, size_t bytesSent);

caerDeviceHandle davisFX3Open(uint16_t deviceID, uint8_t busNumberRestrict, uint8_t devAddressRestrict,
	const char *serialNumberRestrict) {
	caerLog(CAER_LOG_DEBUG, __func__, "Initializing %s.", DAVIS_FX3_DEVICE_NAME);

	davisFX3Handle handle = calloc(1, sizeof(*handle));
	if (handle == NULL) {
		// Failed to allocate memory for device handle!
		caerLog(CAER_LOG_CRITICAL, __func__, "Failed to allocate memory for device handle.");
		return (NULL);
	}

	// Set main deviceType correctly right away.
	handle->h.deviceType = CAER_DEVICE_DAVIS_FX3;

	bool openRetVal = davisCommonOpen((davisHandle) handle, USB_DEFAULT_DEVICE_VID, DAVIS_FX3_DEVICE_PID,
	DAVIS_FX3_DEVICE_NAME, deviceID, busNumberRestrict, devAddressRestrict, serialNumberRestrict,
	DAVIS_FX3_REQUIRED_LOGIC_REVISION, DAVIS_FX3_REQUIRED_FIRMWARE_VERSION);
	if (!openRetVal) {
		free(handle);

		// Failed to open device and grab basic information!
		return (NULL);
	}

	allocateDebugTransfers(handle);

	return ((caerDeviceHandle) handle);
}

bool davisFX3Close(caerDeviceHandle cdh) {
	caerLog(CAER_LOG_DEBUG, ((davisHandle) cdh)->info.deviceString, "Shutting down ...");

	cancelAndDeallocateDebugTransfers((davisFX3Handle) cdh);

	return (davisCommonClose((davisHandle) cdh));
}

bool davisFX3SendDefaultConfig(caerDeviceHandle cdh) {
	// First send default chip/bias config.
	if (!davisCommonSendDefaultChipConfig(cdh, &davisFX3ConfigSet)) {
		return (false);
	}

	// Send default FPGA config.
	if (!davisCommonSendDefaultFPGAConfig(cdh, &davisFX3ConfigSet)) {
		return (false);
	}

	return (true);
}

bool davisFX3ConfigSet(caerDeviceHandle cdh, int8_t modAddr, uint8_t paramAddr, uint32_t param) {
	davisHandle handle = (davisHandle) cdh;

	return (davisCommonConfigSet(handle, modAddr, paramAddr, param));
}

bool davisFX3ConfigGet(caerDeviceHandle cdh, int8_t modAddr, uint8_t paramAddr, uint32_t *param) {
	davisHandle handle = (davisHandle) cdh;

	return (davisCommonConfigGet(handle, modAddr, paramAddr, param));
}

static void allocateDebugTransfers(davisFX3Handle handle) {
	// Set number of transfers and allocate memory for the main transfer array.

	// Allocate transfers and set them up.
	for (size_t i = 0; i < DEBUG_TRANSFER_NUM; i++) {
		handle->debugTransfers[i] = libusb_alloc_transfer(0);
		if (handle->debugTransfers[i] == NULL) {
			caerLog(CAER_LOG_CRITICAL, handle->h.info.deviceString,
				"Unable to allocate further libusb transfers (debug channel, %zu of %" PRIu32 ").", i,
				DEBUG_TRANSFER_NUM);
			continue;
		}

		// Create data buffer.
		handle->debugTransfers[i]->length = DEBUG_TRANSFER_SIZE;
		handle->debugTransfers[i]->buffer = malloc(DEBUG_TRANSFER_SIZE);
		if (handle->debugTransfers[i]->buffer == NULL) {
			caerLog(CAER_LOG_CRITICAL, handle->h.info.deviceString,
				"Unable to allocate buffer for libusb transfer %zu (debug channel). Error: %d.", i, errno);

			libusb_free_transfer(handle->debugTransfers[i]);
			handle->debugTransfers[i] = NULL;

			continue;
		}

		// Initialize Transfer.
		handle->debugTransfers[i]->dev_handle = handle->h.state.usbState.deviceHandle;
		handle->debugTransfers[i]->endpoint = DEBUG_ENDPOINT;
		handle->debugTransfers[i]->type = LIBUSB_TRANSFER_TYPE_INTERRUPT;
		handle->debugTransfers[i]->callback = &libUsbDebugCallback;
		handle->debugTransfers[i]->user_data = handle;
		handle->debugTransfers[i]->timeout = 0;
		handle->debugTransfers[i]->flags = LIBUSB_TRANSFER_FREE_BUFFER;

		if ((errno = libusb_submit_transfer(handle->debugTransfers[i])) == LIBUSB_SUCCESS) {
			atomic_fetch_add(&handle->activeDebugTransfers, 1);
		}
		else {
			caerLog(CAER_LOG_CRITICAL, handle->h.info.deviceString,
				"Unable to submit libusb transfer %zu (debug channel). Error: %s (%d).", i, libusb_strerror(errno),
				errno);

			// The transfer buffer is freed automatically here thanks to
			// the LIBUSB_TRANSFER_FREE_BUFFER flag set above.
			libusb_free_transfer(handle->debugTransfers[i]);
			handle->debugTransfers[i] = NULL;

			continue;
		}
	}

	if (atomic_load(&handle->activeDebugTransfers) == 0) {
		// Didn't manage to allocate any USB transfers, log failure.
		caerLog(CAER_LOG_CRITICAL, handle->h.info.deviceString,
			"Unable to allocate any libusb transfers (debug channel).");
	}
}

static inline void cancelDebugTransfers(davisFX3Handle handle) {
	// Cancel all current transfers.
	for (size_t i = 0; i < DEBUG_TRANSFER_NUM; i++) {
		if (handle->debugTransfers[i] != NULL) {
			errno = libusb_cancel_transfer(handle->debugTransfers[i]);
			if (errno != LIBUSB_SUCCESS && errno != LIBUSB_ERROR_NOT_FOUND) {
				caerLog(CAER_LOG_CRITICAL, handle->h.info.deviceString,
					"Unable to cancel libusb transfer %zu (debug channel). Error: %s (%d).", i, libusb_strerror(errno),
					errno);
				// Proceed with trying to cancel all transfers regardless of errors.
			}
		}
	}
}

static void cancelAndDeallocateDebugTransfers(davisFX3Handle handle) {
	// Cancel all transfers.
	cancelDebugTransfers(handle);

	// Wait for all transfers to go away.
	struct timespec waitForTerminationSleep = { .tv_sec = 0, .tv_nsec = 1000000 };

	while (atomic_load(&handle->activeDebugTransfers) > 0) {
		// Sleep for 1ms to avoid busy loop.
		thrd_sleep(&waitForTerminationSleep, NULL);

		// Continue trying to cancel all transfers until there are none left.
		// It seems like the first cancel pass is not enough and some hang around.
		cancelDebugTransfers(handle);
	}

	// No more transfers in flight, deallocate them all here.
	for (size_t i = 0; i < DEBUG_TRANSFER_NUM; i++) {
		if (handle->debugTransfers[i] != NULL) {
			libusb_free_transfer(handle->debugTransfers[i]);
			handle->debugTransfers[i] = NULL;
		}
	}
}

static void LIBUSB_CALL libUsbDebugCallback(struct libusb_transfer *transfer) {
	davisFX3Handle handle = transfer->user_data;

	if (transfer->status == LIBUSB_TRANSFER_COMPLETED) {
		// Handle debug data.
		debugTranslator(handle, transfer->buffer, (size_t) transfer->actual_length);
	}

	if (transfer->status != LIBUSB_TRANSFER_CANCELLED && transfer->status != LIBUSB_TRANSFER_NO_DEVICE) {
		// Submit transfer again.
		if (libusb_submit_transfer(transfer) == LIBUSB_SUCCESS) {
			return;
		}
	}

	// Cannot recover (cancelled, no device, or other critical error).
	// Signal this by adjusting the counter and exiting.
	// Freeing the transfers is taken care of by cancelAndDeallocateDebugTransfers().
	atomic_fetch_sub(&handle->activeDebugTransfers, 1);
}

static void debugTranslator(davisFX3Handle handle, uint8_t *buffer, size_t bytesSent) {
	// Check if this is a debug message (length 7-64 bytes).
	if (bytesSent >= 7 && buffer[0] == 0x00) {
		// Debug message, log this.
		caerLog(CAER_LOG_ERROR, handle->h.info.deviceString, "Error message: '%s' (code %u at time %u).", &buffer[6],
			buffer[1], *((uint32_t *) &buffer[2]));
	}
	else {
		// Unknown/invalid debug message, log this.
		caerLog(CAER_LOG_WARNING, handle->h.info.deviceString, "Unknown/invalid debug message.");
	}
}
