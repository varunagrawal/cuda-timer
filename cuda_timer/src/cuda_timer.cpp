#pragma once

#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include <torch/torch.h>

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define gpuErrorCheck(ans)                    \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

int64_t cuda_timer_start()
{
    cudaEvent_t event;
    gpuErrorCheck(cudaEventCreate(&event));
    //gpuErrorCheck(cudaEventRecord(event, stream ? *stream : 0));
    gpuErrorCheck(cudaEventRecord(event));
    int64_t output = reinterpret_cast<int64_t>(event);
    return output;
}

float cuda_timer_end(int64_t start_event)
{
    // Create and record the end event
    cudaEvent_t end_event;
    gpuErrorCheck(cudaEventCreate(&end_event));

    gpuErrorCheck(cudaEventRecord(end_event)); //, stream ? *stream : 0));
    gpuErrorCheck(cudaEventSynchronize(end_event));

    // Get the time difference
    cudaEvent_t startEvent = reinterpret_cast<cudaEvent_t>(start_event);

    //printf("startEvent handle=%x\n", startEvent);
    float time;
    gpuErrorCheck(cudaEventElapsedTime(&time, startEvent, end_event)); // ms
    //printf("TIME=%.2f\n", time);
    gpuErrorCheck(cudaEventDestroy(startEvent));

    //printf("Destroying end event\n");
    gpuErrorCheck(cudaEventDestroy(end_event));

    return time;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("cuda_timer_start", &cuda_timer_start, "Start CUDA Timer");
    m.def("cuda_timer_end", &cuda_timer_end, "Stop CUDA Timer");
}