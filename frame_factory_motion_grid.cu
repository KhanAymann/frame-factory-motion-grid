#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << cudaGetErrorString(err) << std::endl;                       \
      std::exit(1);                                                             \
    }                                                                          \
  } while (0)

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr int kChannels = 3;
constexpr int kCameraCount = 3;
constexpr int kMotionThreshold = 20;
constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkIterations = 60;
constexpr int kThreadsPerBlock = 256;

double nowMs() {
  using clock = std::chrono::high_resolution_clock;
  static const auto start = clock::now();

  auto current = clock::now();
  std::chrono::duration<double, std::milli> elapsed = current - start;

  return elapsed.count();
}

uint8_t grayFromRgbCpu(const uint8_t* rgb, int rgbIndex) {
  int r = static_cast<int>(rgb[rgbIndex + 0]);
  int g = static_cast<int>(rgb[rgbIndex + 1]);
  int b = static_cast<int>(rgb[rgbIndex + 2]);

  return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

__device__ __forceinline__ uint8_t grayFromRgbGpu(
    const uint8_t* rgb,
    int rgbIndex
) {
  int r = static_cast<int>(rgb[rgbIndex + 0]);
  int g = static_cast<int>(rgb[rgbIndex + 1]);
  int b = static_cast<int>(rgb[rgbIndex + 2]);

  return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

__global__ void motionMapKernel(
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    uint8_t* motionMask,
    unsigned int* motionPixelCount,
    int totalPixels,
    int threshold
) {
  extern __shared__ unsigned int blockCounts[];

  int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int localCount = 0;

  if (pixelIndex < totalPixels) {
    int rgbIndex = pixelIndex * 3;

    uint8_t previousGray = grayFromRgbGpu(previousRgb, rgbIndex);
    uint8_t currentGray = grayFromRgbGpu(currentRgb, rgbIndex);

    int diff = static_cast<int>(currentGray) - static_cast<int>(previousGray);
    if (diff < 0) {
      diff = -diff;
    }

    uint8_t moving = diff > threshold ? 255 : 0;
    motionMask[pixelIndex] = moving;
    localCount = moving ? 1 : 0;
  }

  blockCounts[threadIdx.x] = localCount;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      blockCounts[threadIdx.x] += blockCounts[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomicAdd(motionPixelCount, blockCounts[0]);
  }
}

void fillFakeMotionFrames(uint8_t* previousRgb, uint8_t* currentRgb) {
  const int pixelsPerCamera = kWidth * kHeight;
  const int objectWidth = 360;
  const int objectHeight = 220;

  for (int camera = 0; camera < kCameraCount; camera++) {
    int previousX = 260 + camera * 120;
    int previousY = 260 + camera * 85;
    int currentX = previousX + 28 + camera * 4;
    int currentY = previousY + 18 + camera * 3;

    for (int y = 0; y < kHeight; y++) {
      for (int x = 0; x < kWidth; x++) {
        int pixelIndex = camera * pixelsPerCamera + y * kWidth + x;
        int rgbIndex = pixelIndex * kChannels;

        uint8_t r = static_cast<uint8_t>((x * 3 + y * 5 + camera * 23) % 160);
        uint8_t g = static_cast<uint8_t>((x * 7 + y * 2 + camera * 41) % 160);
        uint8_t b = static_cast<uint8_t>((x * 2 + y * 9 + camera * 67) % 160);

        previousRgb[rgbIndex + 0] = r;
        previousRgb[rgbIndex + 1] = g;
        previousRgb[rgbIndex + 2] = b;

        currentRgb[rgbIndex + 0] = r;
        currentRgb[rgbIndex + 1] = g;
        currentRgb[rgbIndex + 2] = b;

        bool inPreviousObject =
            x >= previousX && x < previousX + objectWidth &&
            y >= previousY && y < previousY + objectHeight;
        bool inCurrentObject =
            x >= currentX && x < currentX + objectWidth &&
            y >= currentY && y < currentY + objectHeight;

        if (inPreviousObject) {
          previousRgb[rgbIndex + 0] = static_cast<uint8_t>(r + 70);
          previousRgb[rgbIndex + 1] = static_cast<uint8_t>(g + 45);
          previousRgb[rgbIndex + 2] = static_cast<uint8_t>(b + 25);
        }

        if (inCurrentObject) {
          currentRgb[rgbIndex + 0] = static_cast<uint8_t>(r + 70);
          currentRgb[rgbIndex + 1] = static_cast<uint8_t>(g + 45);
          currentRgb[rgbIndex + 2] = static_cast<uint8_t>(b + 25);
        }
      }
    }
  }
}

unsigned int runCpuMotionStep(
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    uint8_t* previousGray,
    uint8_t* currentGray,
    uint8_t* motionMask,
    int totalPixels
) {
  for (int pixel = 0; pixel < totalPixels; pixel++) {
    int rgbIndex = pixel * kChannels;
    previousGray[pixel] = grayFromRgbCpu(previousRgb, rgbIndex);
    currentGray[pixel] = grayFromRgbCpu(currentRgb, rgbIndex);
  }

  unsigned int motionPixelCount = 0;

  for (int pixel = 0; pixel < totalPixels; pixel++) {
    int diff =
        static_cast<int>(currentGray[pixel]) - static_cast<int>(previousGray[pixel]);
    if (diff < 0) {
      diff = -diff;
    }

    uint8_t moving = diff > kMotionThreshold ? 255 : 0;
    motionMask[pixel] = moving;
    motionPixelCount += moving ? 1 : 0;
  }

  return motionPixelCount;
}

struct CpuResult {
  double averageMs;
  double fps;
  unsigned int motionPixelCount;
};

CpuResult runCpuBaseline(
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    int totalPixels
) {
  std::vector<uint8_t> previousGray(totalPixels);
  std::vector<uint8_t> currentGray(totalPixels);
  std::vector<uint8_t> motionMask(totalPixels);

  unsigned int motionPixelCount = 0;

  for (int i = 0; i < kWarmupIterations; i++) {
    motionPixelCount = runCpuMotionStep(
        previousRgb,
        currentRgb,
        previousGray.data(),
        currentGray.data(),
        motionMask.data(),
        totalPixels
    );
  }

  double start = nowMs();

  for (int i = 0; i < kBenchmarkIterations; i++) {
    motionPixelCount = runCpuMotionStep(
        previousRgb,
        currentRgb,
        previousGray.data(),
        currentGray.data(),
        motionMask.data(),
        totalPixels
    );
  }

  double end = nowMs();
  double averageMs = (end - start) / kBenchmarkIterations;

  return {averageMs, 1000.0 / averageMs, motionPixelCount};
}

struct GpuResult {
  std::string name;
  double hostToDeviceMs;
  double kernelMs;
  double deviceToHostMs;
  double totalMs;
  double fps;
  double speedupVsCpu;
  unsigned int motionPixelCount;
};

GpuResult runGpuBenchmark(
    const std::string& name,
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    int totalPixels,
    size_t rgbBytes,
    size_t maskBytes,
    bool pinnedHostCount,
    double cpuAverageMs
) {
  uint8_t* gpuPreviousRgb = nullptr;
  uint8_t* gpuCurrentRgb = nullptr;
  uint8_t* gpuMotionMask = nullptr;
  unsigned int* gpuMotionCount = nullptr;

  CUDA_CHECK(cudaMalloc(&gpuPreviousRgb, rgbBytes));
  CUDA_CHECK(cudaMalloc(&gpuCurrentRgb, rgbBytes));
  CUDA_CHECK(cudaMalloc(&gpuMotionMask, maskBytes));
  CUDA_CHECK(cudaMalloc(&gpuMotionCount, sizeof(unsigned int)));

  unsigned int pageableMotionCount = 0;
  unsigned int* hostMotionCount = &pageableMotionCount;

  if (pinnedHostCount) {
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&hostMotionCount),
        sizeof(unsigned int)
    ));
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t h2dStart = nullptr;
  cudaEvent_t h2dStop = nullptr;
  cudaEvent_t kernelStart = nullptr;
  cudaEvent_t kernelStop = nullptr;
  cudaEvent_t d2hStart = nullptr;
  cudaEvent_t d2hStop = nullptr;

  CUDA_CHECK(cudaStreamCreate(&stream));
  CUDA_CHECK(cudaEventCreate(&h2dStart));
  CUDA_CHECK(cudaEventCreate(&h2dStop));
  CUDA_CHECK(cudaEventCreate(&kernelStart));
  CUDA_CHECK(cudaEventCreate(&kernelStop));
  CUDA_CHECK(cudaEventCreate(&d2hStart));
  CUDA_CHECK(cudaEventCreate(&d2hStop));

  int blocks = (totalPixels + kThreadsPerBlock - 1) / kThreadsPerBlock;
  size_t sharedBytes = kThreadsPerBlock * sizeof(unsigned int);

  for (int i = 0; i < kWarmupIterations; i++) {
    CUDA_CHECK(cudaMemcpyAsync(
        gpuPreviousRgb,
        previousRgb,
        rgbBytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        gpuCurrentRgb,
        currentRgb,
        rgbBytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK(cudaMemsetAsync(gpuMotionCount, 0, sizeof(unsigned int), stream));

    motionMapKernel<<<blocks, kThreadsPerBlock, sharedBytes, stream>>>(
        gpuPreviousRgb,
        gpuCurrentRgb,
        gpuMotionMask,
        gpuMotionCount,
        totalPixels,
        kMotionThreshold
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(
        hostMotionCount,
        gpuMotionCount,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  double totalHostToDeviceMs = 0.0;
  double totalKernelMs = 0.0;
  double totalDeviceToHostMs = 0.0;
  double totalWallMs = 0.0;

  for (int i = 0; i < kBenchmarkIterations; i++) {
    *hostMotionCount = 0;

    double wallStart = nowMs();

    CUDA_CHECK(cudaEventRecord(h2dStart, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        gpuPreviousRgb,
        previousRgb,
        rgbBytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK(cudaMemcpyAsync(
        gpuCurrentRgb,
        currentRgb,
        rgbBytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    CUDA_CHECK(cudaEventRecord(h2dStop, stream));

    CUDA_CHECK(cudaMemsetAsync(gpuMotionCount, 0, sizeof(unsigned int), stream));

    CUDA_CHECK(cudaEventRecord(kernelStart, stream));
    motionMapKernel<<<blocks, kThreadsPerBlock, sharedBytes, stream>>>(
        gpuPreviousRgb,
        gpuCurrentRgb,
        gpuMotionMask,
        gpuMotionCount,
        totalPixels,
        kMotionThreshold
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernelStop, stream));

    CUDA_CHECK(cudaEventRecord(d2hStart, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        hostMotionCount,
        gpuMotionCount,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaEventRecord(d2hStop, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    double wallEnd = nowMs();
    totalWallMs += wallEnd - wallStart;

    float h2dMs = 0.0f;
    float kernelMs = 0.0f;
    float d2hMs = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(&h2dMs, h2dStart, h2dStop));
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, kernelStart, kernelStop));
    CUDA_CHECK(cudaEventElapsedTime(&d2hMs, d2hStart, d2hStop));

    totalHostToDeviceMs += static_cast<double>(h2dMs);
    totalKernelMs += static_cast<double>(kernelMs);
    totalDeviceToHostMs += static_cast<double>(d2hMs);
  }

  unsigned int motionPixelCount = *hostMotionCount;

  CUDA_CHECK(cudaEventDestroy(h2dStart));
  CUDA_CHECK(cudaEventDestroy(h2dStop));
  CUDA_CHECK(cudaEventDestroy(kernelStart));
  CUDA_CHECK(cudaEventDestroy(kernelStop));
  CUDA_CHECK(cudaEventDestroy(d2hStart));
  CUDA_CHECK(cudaEventDestroy(d2hStop));
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (pinnedHostCount) {
    CUDA_CHECK(cudaFreeHost(hostMotionCount));
  }

  CUDA_CHECK(cudaFree(gpuPreviousRgb));
  CUDA_CHECK(cudaFree(gpuCurrentRgb));
  CUDA_CHECK(cudaFree(gpuMotionMask));
  CUDA_CHECK(cudaFree(gpuMotionCount));

  double averageTotalMs = totalWallMs / kBenchmarkIterations;

  return {
      name,
      totalHostToDeviceMs / kBenchmarkIterations,
      totalKernelMs / kBenchmarkIterations,
      totalDeviceToHostMs / kBenchmarkIterations,
      averageTotalMs,
      1000.0 / averageTotalMs,
      cpuAverageMs / averageTotalMs,
      motionPixelCount
  };
}

void printGpuResult(const GpuResult& result) {
  std::cout << result.name << "\n";
  std::cout << "  CPU -> GPU copy:     " << result.hostToDeviceMs << " ms\n";
  std::cout << "  Kernel time:         " << result.kernelMs << " ms\n";
  std::cout << "  GPU -> CPU copy:     " << result.deviceToHostMs << " ms\n";
  std::cout << "  Total time:          " << result.totalMs << " ms\n";
  std::cout << "  Motion pixel count:  " << result.motionPixelCount << "\n";
  std::cout << "  Estimated FPS:       " << result.fps << "\n";
  std::cout << "  Speedup vs CPU:      " << result.speedupVsCpu << "x\n\n";
}

void printCpuResult(const CpuResult& result) {
  std::cout << "CPU baseline\n";
  std::cout << "  CPU time:            " << result.averageMs << " ms\n";
  std::cout << "  Motion pixel count:  " << result.motionPixelCount << "\n";
  std::cout << "  Estimated FPS:       " << result.fps << "\n\n";
}

std::string yesNo(bool value) {
  return value ? "yes" : "no";
}

int main() {
  const int pixelsPerCamera = kWidth * kHeight;
  const int totalPixels = pixelsPerCamera * kCameraCount;

  const size_t rgbBytes = static_cast<size_t>(totalPixels) * kChannels;
  const size_t totalInputBytes = rgbBytes * 2;
  const size_t maskBytes = static_cast<size_t>(totalPixels);
  const double totalInputMB = static_cast<double>(totalInputBytes) / 1000000.0;
  const double totalInputMiB =
      static_cast<double>(totalInputBytes) / (1024.0 * 1024.0);

  std::cout << std::fixed << std::setprecision(3);

  std::cout << "Frame Factory 2: Motion Map\n";
  std::cout << "Previous/current RGB frame difference benchmark\n\n";

  std::cout << "Cameras: " << kCameraCount << "\n";
  std::cout << "Frame dimensions: " << kWidth << "x" << kHeight << " RGB\n";
  std::cout << "Pixels processed: " << totalPixels << "\n";
  std::cout << "Motion threshold: " << kMotionThreshold
            << " grayscale levels\n";
  std::cout << "Total input bytes (previous + current): " << totalInputBytes
            << " bytes (" << totalInputMB << " MB / " << totalInputMiB
            << " MiB)\n";
  std::cout << "GPU result copied back: " << sizeof(unsigned int)
            << " bytes motion count\n";
  std::cout << "Benchmark iterations: " << kBenchmarkIterations << "\n\n";

  std::vector<uint8_t> pageablePreviousRgb(rgbBytes);
  std::vector<uint8_t> pageableCurrentRgb(rgbBytes);

  fillFakeMotionFrames(pageablePreviousRgb.data(), pageableCurrentRgb.data());

  CpuResult cpuResult = runCpuBaseline(
      pageablePreviousRgb.data(),
      pageableCurrentRgb.data(),
      totalPixels
  );

  printCpuResult(cpuResult);

  int cudaDeviceCount = 0;
  cudaError_t deviceStatus = cudaGetDeviceCount(&cudaDeviceCount);
  if (deviceStatus != cudaSuccess || cudaDeviceCount == 0) {
    std::cerr << "GPU benchmark skipped: "
              << cudaGetErrorString(deviceStatus) << "\n";
    return 1;
  }

  CUDA_CHECK(cudaSetDevice(0));

  uint8_t* pinnedPreviousRgb = nullptr;
  uint8_t* pinnedCurrentRgb = nullptr;

  CUDA_CHECK(cudaMallocHost(
      reinterpret_cast<void**>(&pinnedPreviousRgb),
      rgbBytes
  ));
  CUDA_CHECK(cudaMallocHost(
      reinterpret_cast<void**>(&pinnedCurrentRgb),
      rgbBytes
  ));

  std::memcpy(pinnedPreviousRgb, pageablePreviousRgb.data(), rgbBytes);
  std::memcpy(pinnedCurrentRgb, pageableCurrentRgb.data(), rgbBytes);

  GpuResult pageableResult = runGpuBenchmark(
      "GPU pageable memory",
      pageablePreviousRgb.data(),
      pageableCurrentRgb.data(),
      totalPixels,
      rgbBytes,
      maskBytes,
      false,
      cpuResult.averageMs
  );

  GpuResult pinnedResult = runGpuBenchmark(
      "GPU pinned memory",
      pinnedPreviousRgb,
      pinnedCurrentRgb,
      totalPixels,
      rgbBytes,
      maskBytes,
      true,
      cpuResult.averageMs
  );

  CUDA_CHECK(cudaFreeHost(pinnedPreviousRgb));
  CUDA_CHECK(cudaFreeHost(pinnedCurrentRgb));

  printGpuResult(pageableResult);
  printGpuResult(pinnedResult);

  double pinnedVsPageable = pageableResult.totalMs / pinnedResult.totalMs;

  std::cout << "Comparison\n";
  std::cout << "  Pinned vs pageable total speedup: " << pinnedVsPageable << "x\n";
  std::cout << "  Pageable count matches CPU: "
            << yesNo(pageableResult.motionPixelCount == cpuResult.motionPixelCount)
            << "\n";
  std::cout << "  Pinned count matches CPU: "
            << yesNo(pinnedResult.motionPixelCount == cpuResult.motionPixelCount)
            << "\n\n";

  std::cout << "30 FPS frame budget: 33.33 ms\n";
  std::cout << "60 FPS frame budget: 16.67 ms\n";

  return 0;
}
