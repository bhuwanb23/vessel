// =============================================================================
// nvml_loader.cpp — Dynamic NVML loader implementation
// =============================================================================

#include "nvml_loader.h"
#include <cstdio>

// --- Global function pointers ---

void*          nvml_dll_handle = nullptr;
pfn_nvmlInit   nvml_fn_Init = nullptr;
pfn_nvmlShutdown nvml_fn_Shutdown = nullptr;
pfn_nvmlErrorString nvml_fn_ErrorString = nullptr;
pfn_nvmlDeviceGetCount nvml_fn_DeviceGetCount = nullptr;
pfn_nvmlDeviceGetHandleByIndex nvml_fn_DeviceGetHandleByIndex = nullptr;
pfn_nvmlDeviceGetName nvml_fn_DeviceGetName = nullptr;
pfn_nvmlDeviceGetCudaComputeCapability nvml_fn_DeviceGetCudaComputeCapability = nullptr;
pfn_nvmlDeviceGetMemoryInfo nvml_fn_DeviceGetMemoryInfo = nullptr;
pfn_nvmlDeviceGetTemperature nvml_fn_DeviceGetTemperature = nullptr;
pfn_nvmlDeviceGetClockInfo nvml_fn_DeviceGetClockInfo = nullptr;
pfn_nvmlDeviceGetMaxClockInfo nvml_fn_DeviceGetMaxClockInfo = nullptr;
pfn_nvmlDeviceGetUtilizationRates nvml_fn_DeviceGetUtilizationRates = nullptr;
pfn_nvmlDeviceGetPcieThroughput nvml_fn_DeviceGetPcieThroughput = nullptr;

// --- Platform-specific dynamic loading ---

#ifdef _WIN32
#define LOAD_LIBRARY(name) LoadLibraryA(name)
#define GET_PROC(handle, name) GetProcAddress((HMODULE)handle, name)
#define FREE_LIBRARY(handle) FreeLibrary((HMODULE)handle)
#define NVML_LIB_NAME "nvml.dll"
#else
#define LOAD_LIBRARY(name) dlopen(name, RTLD_NOW)
#define GET_PROC(handle, name) dlsym(handle, name)
#define FREE_LIBRARY(handle) dlclose(handle)
#ifdef __APPLE__
#define NVML_LIB_NAME "libnvidia-ml.dylib"
#else
#define NVML_LIB_NAME "libnvidia-ml.so.1"
#endif
#endif

// Helper to load a symbol and report failure
static void* load_sym(void* handle, const char* name) {
    void* sym = GET_PROC(handle, name);
    if (!sym) {
        fprintf(stderr, "[NVML] Warning: symbol '%s' not found\n", name);
    }
    return sym;
}

bool nvml_loader_init() {
    if (nvml_dll_handle) return true;  // already loaded

    nvml_dll_handle = LOAD_LIBRARY(NVML_LIB_NAME);
    if (!nvml_dll_handle) {
        // Try alternate names on Linux
#ifndef _WIN32
        nvml_dll_handle = LOAD_LIBRARY("libnvidia-ml.so");
#endif
    }
    if (!nvml_dll_handle) {
        fprintf(stderr, "[NVML] Library not found — GPU monitoring disabled\n");
        return false;
    }

    // Load all required symbols
    // nvmlInit may be nvmlInit_v2 (preferred) or nvmlInit (legacy)
    nvml_fn_Init = (pfn_nvmlInit)GET_PROC(nvml_dll_handle, "nvmlInit_v2");
    if (!nvml_fn_Init) nvml_fn_Init = (pfn_nvmlInit)GET_PROC(nvml_dll_handle, "nvmlInit");
    nvml_fn_Shutdown = (pfn_nvmlShutdown)GET_PROC(nvml_dll_handle, "nvmlShutdown");
    nvml_fn_ErrorString = (pfn_nvmlErrorString)GET_PROC(nvml_dll_handle, "nvmlErrorString");
    nvml_fn_DeviceGetCount = (pfn_nvmlDeviceGetCount)GET_PROC(nvml_dll_handle, "nvmlDeviceGetCount");
    // nvmlDeviceGetHandleByIndex may be _v2 (preferred) or plain
    nvml_fn_DeviceGetHandleByIndex = (pfn_nvmlDeviceGetHandleByIndex)GET_PROC(nvml_dll_handle, "nvmlDeviceGetHandleByIndex_v2");
    if (!nvml_fn_DeviceGetHandleByIndex) nvml_fn_DeviceGetHandleByIndex = (pfn_nvmlDeviceGetHandleByIndex)GET_PROC(nvml_dll_handle, "nvmlDeviceGetHandleByIndex");
    nvml_fn_DeviceGetName = (pfn_nvmlDeviceGetName)GET_PROC(nvml_dll_handle, "nvmlDeviceGetName");
    nvml_fn_DeviceGetCudaComputeCapability = (pfn_nvmlDeviceGetCudaComputeCapability)GET_PROC(nvml_dll_handle, "nvmlDeviceGetCudaComputeCapability");
    nvml_fn_DeviceGetMemoryInfo = (pfn_nvmlDeviceGetMemoryInfo)GET_PROC(nvml_dll_handle, "nvmlDeviceGetMemoryInfo");
    nvml_fn_DeviceGetTemperature = (pfn_nvmlDeviceGetTemperature)GET_PROC(nvml_dll_handle, "nvmlDeviceGetTemperature");
    nvml_fn_DeviceGetClockInfo = (pfn_nvmlDeviceGetClockInfo)GET_PROC(nvml_dll_handle, "nvmlDeviceGetClockInfo");
    nvml_fn_DeviceGetMaxClockInfo = (pfn_nvmlDeviceGetMaxClockInfo)GET_PROC(nvml_dll_handle, "nvmlDeviceGetMaxClockInfo");
    nvml_fn_DeviceGetUtilizationRates = (pfn_nvmlDeviceGetUtilizationRates)GET_PROC(nvml_dll_handle, "nvmlDeviceGetUtilizationRates");
    nvml_fn_DeviceGetPcieThroughput = (pfn_nvmlDeviceGetPcieThroughput)GET_PROC(nvml_dll_handle, "nvmlDeviceGetPcieThroughput");

    // Check that critical symbols are available
    if (!nvml_fn_Init || !nvml_fn_DeviceGetCount || !nvml_fn_DeviceGetHandleByIndex) {
        fprintf(stderr, "[NVML] Critical symbols missing — GPU monitoring disabled\n");
        FREE_LIBRARY(nvml_dll_handle);
        nvml_dll_handle = nullptr;
        return false;
    }

    return true;
}

void nvml_loader_shutdown() {
    if (nvml_dll_handle) {
        if (nvml_fn_Shutdown) nvml_fn_Shutdown();
        FREE_LIBRARY(nvml_dll_handle);
        nvml_dll_handle = nullptr;
    }
}
