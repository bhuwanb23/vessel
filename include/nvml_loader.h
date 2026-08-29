#pragma once
// =============================================================================
// nvml_loader.h — Dynamic NVML loader (no nvml.lib required at link time)
//
// Loads NVIDIA Management Library at runtime via LoadLibrary/dlopen.
// All NVML types are defined here so source files don't need <nvml.h>.
// =============================================================================

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdint>

// --- NVML Types (subset used by this project) ---

typedef int nvmlReturn_t;
typedef void* nvmlDevice_t;

typedef struct {
    uint64_t total;
    uint64_t free;
    uint64_t used;
} nvmlMemory_t;

typedef struct {
    uint32_t gpu;
    uint32_t memory;
} nvmlUtilization_t;

// --- NVML Constants ---

#define NVML_SUCCESS 0
#define NVML_TEMPERATURE_GPU 0
#define NVML_CLOCK_SM 0
#define NVML_CLOCK_MEM 1
#define NVML_PCIE_UTIL_RX_BYTES 1
#define NVML_PCIE_UTIL_TX_BYTES 2
#define NVML_ERROR_LENGTH 96
#define NVML_DEVICE_NAME_V2_BUFFER_SIZE 96

// --- Function pointer types ---

typedef nvmlReturn_t (*pfn_nvmlInit)(void);
typedef nvmlReturn_t (*pfn_nvmlShutdown)(void);
typedef const char*  (*pfn_nvmlErrorString)(nvmlReturn_t);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCount)(unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetCudaComputeCapability)(nvmlDevice_t, int*, int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetTemperature)(nvmlDevice_t, unsigned int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetClockInfo)(nvmlDevice_t, unsigned int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetMaxClockInfo)(nvmlDevice_t, unsigned int, unsigned int*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t*);
typedef nvmlReturn_t (*pfn_nvmlDeviceGetPcieThroughput)(nvmlDevice_t, unsigned int, unsigned int*);

// --- Extern function pointers (defined in nvml_loader.cpp) ---

extern void*          nvml_dll_handle;
extern pfn_nvmlInit   nvml_fn_Init;
extern pfn_nvmlShutdown nvml_fn_Shutdown;
extern pfn_nvmlErrorString nvml_fn_ErrorString;
extern pfn_nvmlDeviceGetCount nvml_fn_DeviceGetCount;
extern pfn_nvmlDeviceGetHandleByIndex nvml_fn_DeviceGetHandleByIndex;
extern pfn_nvmlDeviceGetName nvml_fn_DeviceGetName;
extern pfn_nvmlDeviceGetCudaComputeCapability nvml_fn_DeviceGetCudaComputeCapability;
extern pfn_nvmlDeviceGetMemoryInfo nvml_fn_DeviceGetMemoryInfo;
extern pfn_nvmlDeviceGetTemperature nvml_fn_DeviceGetTemperature;
extern pfn_nvmlDeviceGetClockInfo nvml_fn_DeviceGetClockInfo;
extern pfn_nvmlDeviceGetMaxClockInfo nvml_fn_DeviceGetMaxClockInfo;
extern pfn_nvmlDeviceGetUtilizationRates nvml_fn_DeviceGetUtilizationRates;
extern pfn_nvmlDeviceGetPcieThroughput nvml_fn_DeviceGetPcieThroughput;

// --- API ---

// Call once at startup. Returns true if NVML loaded successfully.
bool nvml_loader_init();

// Call at shutdown. Unloads the library.
void nvml_loader_shutdown();

// Check if NVML is loaded and usable
inline bool nvml_loader_is_available() { return nvml_dll_handle != nullptr; }
