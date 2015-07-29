/*
 * dvs128.h
 *
 *  Created on: May 26, 2015
 *      Author: llongi
 */

#ifndef LIBCAER_DEVICES_DVS128_H_
#define LIBCAER_DEVICES_DVS128_H_

#include "usb.h"
#include "events/polarity.h"
#include "events/special.h"

#define CAER_DEVICE_DVS128 0

#define DVS128_CONFIG_DVS  0
#define DVS128_CONFIG_BIAS 1

#define DVS128_CONFIG_DVS_RUN             0
#define DVS128_CONFIG_DVS_TIMESTAMP_RESET 1
#define DVS128_CONFIG_DVS_ARRAY_RESET     2

#define DVS128_CONFIG_BIAS_CAS     0
#define DVS128_CONFIG_BIAS_INJGND  1
#define DVS128_CONFIG_BIAS_PUX     2
#define DVS128_CONFIG_BIAS_PUY     3
#define DVS128_CONFIG_BIAS_REQPD   4
#define DVS128_CONFIG_BIAS_REQ     5
#define DVS128_CONFIG_BIAS_FOLL    6
#define DVS128_CONFIG_BIAS_PR      7
#define DVS128_CONFIG_BIAS_REFR    8
#define DVS128_CONFIG_BIAS_DIFF    9
#define DVS128_CONFIG_BIAS_DIFFON  10
#define DVS128_CONFIG_BIAS_DIFFOFF 11

struct caer_dvs128_info {
	uint16_t deviceID;
	char *deviceString;
	// System information fields
	uint16_t logicVersion;
	bool deviceIsMaster;
	// DVS specific fields
	uint16_t dvsSizeX;
	uint16_t dvsSizeY;
};

typedef struct caer_dvs128_info *caerDVS128Info;

caerDVS128Info caerDVS128InfoGet(caerDeviceHandle handle);

#endif /* LIBCAER_DEVICES_DVS128_H_ */