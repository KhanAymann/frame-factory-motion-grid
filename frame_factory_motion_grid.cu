#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
                << cudaGetErrorString(error) << '\n';                          \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

constexpr int kWidth = 1920;
constexpr int kHeight = 1080;
constexpr int kChannels = 3;
constexpr int kCameraCount = 3;
constexpr int kGridWidth = 16;
constexpr int kGridHeight = 9;
constexpr int kTileWidth = kWidth / kGridWidth;
constexpr int kTileHeight = kHeight / kGridHeight;
constexpr int kTilesPerCamera = kGridWidth * kGridHeight;
constexpr int kTotalTileCounts = kCameraCount * kTilesPerCamera;
constexpr int kPixelsPerCamera = kWidth * kHeight;
constexpr int kTotalPixels = kCameraCount * kPixelsPerCamera;
constexpr int kMotionThreshold = 20;
constexpr int kWarmupIterations = 5;
constexpr int kBenchmarkIterations = 60;
constexpr int kThreadsPerBlock = 256;
constexpr int kPpmTileSize = 48;

static_assert(kWidth % kGridWidth == 0, "Frame width must divide into the grid");
static_assert(
    kHeight % kGridHeight == 0,
    "Frame height must divide into the grid"
);

using MotionGrid = std::array<unsigned int, kTotalTileCounts>;

double nowMs() {
  using Clock = std::chrono::steady_clock;
  static const auto start = Clock::now();
  const std::chrono::duration<double, std::milli> elapsed = Clock::now() - start;
  return elapsed.count();
}

uint8_t grayFromRgbCpu(const uint8_t* rgb, int rgbIndex) {
  const int r = static_cast<int>(rgb[rgbIndex + 0]);
  const int g = static_cast<int>(rgb[rgbIndex + 1]);
  const int b = static_cast<int>(rgb[rgbIndex + 2]);
  return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

__device__ __forceinline__ uint8_t grayFromRgbGpu(
    const uint8_t* rgb,
    int rgbIndex
) {
  const int r = static_cast<int>(rgb[rgbIndex + 0]);
  const int g = static_cast<int>(rgb[rgbIndex + 1]);
  const int b = static_cast<int>(rgb[rgbIndex + 2]);
  return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
}

__global__ void motionGridKernel(
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    unsigned int* motionGrid,
    int totalPixels,
    int threshold
) {
  const int globalPixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalPixel >= totalPixels) {
    return;
  }

  const int rgbIndex = globalPixel * kChannels;
  const uint8_t previousGray = grayFromRgbGpu(previousRgb, rgbIndex);
  const uint8_t currentGray = grayFromRgbGpu(currentRgb, rgbIndex);

  int difference =
      static_cast<int>(currentGray) - static_cast<int>(previousGray);
  if (difference < 0) {
    difference = -difference;
  }

  if (difference <= threshold) {
    return;
  }

  const int camera = globalPixel / kPixelsPerCamera;
  const int pixelInCamera = globalPixel - camera * kPixelsPerCamera;
  const int x = pixelInCamera % kWidth;
  const int y = pixelInCamera / kWidth;
  const int tileX = x / kTileWidth;
  const int tileY = y / kTileHeight;
  const int tileIndex =
      camera * kTilesPerCamera + tileY * kGridWidth + tileX;

  atomicAdd(&motionGrid[tileIndex], 1U);
}

void fillFakeMotionFrames(uint8_t* previousRgb, uint8_t* currentRgb) {
  constexpr int objectWidth = 500;
  constexpr int objectHeight = 340;

  for (int camera = 0; camera < kCameraCount; ++camera) {
    const int previousX = 300 + camera * 140;
    const int previousY = 230 + camera * 70;
    const int currentX = previousX + 170 + camera * 12;
    const int currentY = previousY + 90 + camera;

    for (int y = 0; y < kHeight; ++y) {
      for (int x = 0; x < kWidth; ++x) {
        const int pixelIndex = camera * kPixelsPerCamera + y * kWidth + x;
        const int rgbIndex = pixelIndex * kChannels;

        const uint8_t r =
            static_cast<uint8_t>((x * 3 + y * 5 + camera * 23) % 140);
        const uint8_t g =
            static_cast<uint8_t>((x * 7 + y * 2 + camera * 41) % 140);
        const uint8_t b =
            static_cast<uint8_t>((x * 2 + y * 9 + camera * 67) % 140);

        previousRgb[rgbIndex + 0] = r;
        previousRgb[rgbIndex + 1] = g;
        previousRgb[rgbIndex + 2] = b;
        currentRgb[rgbIndex + 0] = r;
        currentRgb[rgbIndex + 1] = g;
        currentRgb[rgbIndex + 2] = b;

        const bool inPreviousObject =
            x >= previousX && x < previousX + objectWidth &&
            y >= previousY && y < previousY + objectHeight;
        const bool inCurrentObject =
            x >= currentX && x < currentX + objectWidth &&
            y >= currentY && y < currentY + objectHeight;

        if (inPreviousObject) {
          previousRgb[rgbIndex + 0] = static_cast<uint8_t>(r + 80);
          previousRgb[rgbIndex + 1] = static_cast<uint8_t>(g + 55);
          previousRgb[rgbIndex + 2] = static_cast<uint8_t>(b + 35);
        }

        if (inCurrentObject) {
          currentRgb[rgbIndex + 0] = static_cast<uint8_t>(r + 80);
          currentRgb[rgbIndex + 1] = static_cast<uint8_t>(g + 55);
          currentRgb[rgbIndex + 2] = static_cast<uint8_t>(b + 35);
        }
      }
    }
  }
}

void runCpuMotionGridStep(
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    MotionGrid& motionGrid
) {
  motionGrid.fill(0);

  for (int globalPixel = 0; globalPixel < kTotalPixels; ++globalPixel) {
    const int rgbIndex = globalPixel * kChannels;
    const uint8_t previousGray = grayFromRgbCpu(previousRgb, rgbIndex);
    const uint8_t currentGray = grayFromRgbCpu(currentRgb, rgbIndex);

    int difference =
        static_cast<int>(currentGray) - static_cast<int>(previousGray);
    if (difference < 0) {
      difference = -difference;
    }

    if (difference <= kMotionThreshold) {
      continue;
    }

    const int camera = globalPixel / kPixelsPerCamera;
    const int pixelInCamera = globalPixel - camera * kPixelsPerCamera;
    const int x = pixelInCamera % kWidth;
    const int y = pixelInCamera / kWidth;
    const int tileX = x / kTileWidth;
    const int tileY = y / kTileHeight;
    const int tileIndex =
        camera * kTilesPerCamera + tileY * kGridWidth + tileX;
    ++motionGrid[tileIndex];
  }
}

void keepCpuGridObservable(const MotionGrid& motionGrid) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("" : : "g"(motionGrid.data()) : "memory");
#else
  const volatile unsigned int* observableGrid = motionGrid.data();
  (void)observableGrid[0];
#endif
}

struct CpuResult {
  double averageMs;
  double fps;
  MotionGrid motionGrid;
};

CpuResult runCpuBaseline(
    const uint8_t* previousRgb,
    const uint8_t* currentRgb
) {
  MotionGrid motionGrid{};

  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    runCpuMotionGridStep(previousRgb, currentRgb, motionGrid);
    keepCpuGridObservable(motionGrid);
  }

  const double start = nowMs();
  for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
    runCpuMotionGridStep(previousRgb, currentRgb, motionGrid);
    keepCpuGridObservable(motionGrid);
  }
  const double averageMs = (nowMs() - start) / kBenchmarkIterations;

  return {averageMs, 1000.0 / averageMs, motionGrid};
}

struct GpuResult {
  std::string name;
  double hostToDeviceMs;
  double kernelMs;
  double deviceToHostMs;
  double totalMs;
  double fps;
  double speedupVsCpu;
  MotionGrid motionGrid;
};

GpuResult runGpuBenchmark(
    const std::string& name,
    const uint8_t* previousRgb,
    const uint8_t* currentRgb,
    size_t rgbBytes,
    bool usePinnedOutput,
    double cpuAverageMs
) {
  uint8_t* gpuPreviousRgb = nullptr;
  uint8_t* gpuCurrentRgb = nullptr;
  unsigned int* gpuMotionGrid = nullptr;

  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&gpuPreviousRgb),
      rgbBytes
  ));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&gpuCurrentRgb),
      rgbBytes
  ));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&gpuMotionGrid),
      sizeof(unsigned int) * kTotalTileCounts
  ));

  MotionGrid pageableMotionGrid{};
  unsigned int* hostMotionGrid = pageableMotionGrid.data();
  if (usePinnedOutput) {
    CUDA_CHECK(cudaMallocHost(
        reinterpret_cast<void**>(&hostMotionGrid),
        sizeof(unsigned int) * kTotalTileCounts
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

  const int blocks =
      (kTotalPixels + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const size_t gridBytes = sizeof(unsigned int) * kTotalTileCounts;

  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
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
    CUDA_CHECK(cudaMemsetAsync(gpuMotionGrid, 0, gridBytes, stream));

    motionGridKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        gpuPreviousRgb,
        gpuCurrentRgb,
        gpuMotionGrid,
        kTotalPixels,
        kMotionThreshold
    );
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(
        hostMotionGrid,
        gpuMotionGrid,
        gridBytes,
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  double totalHostToDeviceMs = 0.0;
  double totalKernelMs = 0.0;
  double totalDeviceToHostMs = 0.0;
  double totalWallMs = 0.0;

  for (int iteration = 0; iteration < kBenchmarkIterations; ++iteration) {
    const double wallStart = nowMs();

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

    CUDA_CHECK(cudaEventRecord(kernelStart, stream));
    CUDA_CHECK(cudaMemsetAsync(gpuMotionGrid, 0, gridBytes, stream));
    motionGridKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        gpuPreviousRgb,
        gpuCurrentRgb,
        gpuMotionGrid,
        kTotalPixels,
        kMotionThreshold
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernelStop, stream));

    CUDA_CHECK(cudaEventRecord(d2hStart, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        hostMotionGrid,
        gpuMotionGrid,
        gridBytes,
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaEventRecord(d2hStop, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    totalWallMs += nowMs() - wallStart;

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

  MotionGrid resultGrid{};
  std::copy_n(hostMotionGrid, kTotalTileCounts, resultGrid.begin());

  CUDA_CHECK(cudaEventDestroy(h2dStart));
  CUDA_CHECK(cudaEventDestroy(h2dStop));
  CUDA_CHECK(cudaEventDestroy(kernelStart));
  CUDA_CHECK(cudaEventDestroy(kernelStop));
  CUDA_CHECK(cudaEventDestroy(d2hStart));
  CUDA_CHECK(cudaEventDestroy(d2hStop));
  CUDA_CHECK(cudaStreamDestroy(stream));

  if (usePinnedOutput) {
    CUDA_CHECK(cudaFreeHost(hostMotionGrid));
  }
  CUDA_CHECK(cudaFree(gpuPreviousRgb));
  CUDA_CHECK(cudaFree(gpuCurrentRgb));
  CUDA_CHECK(cudaFree(gpuMotionGrid));

  const double averageTotalMs = totalWallMs / kBenchmarkIterations;
  return {
      name,
      totalHostToDeviceMs / kBenchmarkIterations,
      totalKernelMs / kBenchmarkIterations,
      totalDeviceToHostMs / kBenchmarkIterations,
      averageTotalMs,
      1000.0 / averageTotalMs,
      cpuAverageMs / averageTotalMs,
      resultGrid
  };
}

void printCpuResult(const CpuResult& result) {
  std::cout << "CPU baseline:\n";
  std::cout << "  Time:                  " << result.averageMs << " ms\n";
  std::cout << "  FPS:                   " << result.fps << "\n\n";
}

void printGpuResult(const GpuResult& result) {
  std::cout << result.name << ":\n";
  std::cout << "  H2D:                   " << result.hostToDeviceMs << " ms\n";
  std::cout << "  Kernel (clear + grid): " << result.kernelMs << " ms\n";
  std::cout << "  D2H:                   " << result.deviceToHostMs << " ms\n";
  std::cout << "  Total:                 " << result.totalMs << " ms\n";
  std::cout << "  FPS:                   " << result.fps << "\n";
  std::cout << "  Speedup vs CPU:        " << result.speedupVsCpu << "x\n\n";
}

bool validateMotionGrid(
    const MotionGrid& cpuGrid,
    const MotionGrid& gpuGrid,
    const std::string& gpuName
) {
  bool matches = true;

  for (int camera = 0; camera < kCameraCount; ++camera) {
    for (int tileY = 0; tileY < kGridHeight; ++tileY) {
      for (int tileX = 0; tileX < kGridWidth; ++tileX) {
        const int tileIndex =
            camera * kTilesPerCamera + tileY * kGridWidth + tileX;
        if (cpuGrid[tileIndex] == gpuGrid[tileIndex]) {
          continue;
        }

        matches = false;
        std::cerr << "VALIDATION MISMATCH (" << gpuName << ")\n"
                  << "  camera: " << camera << '\n'
                  << "  tile: (" << tileX << ", " << tileY << ")\n"
                  << "  CPU count: " << cpuGrid[tileIndex] << '\n'
                  << "  GPU count: " << gpuGrid[tileIndex] << '\n';
      }
    }
  }

  return matches;
}

char heatmapSymbol(unsigned int motionPixels) {
  if (motionPixels == 0) {
    return '.';
  }

  constexpr unsigned int tilePixels = kTileWidth * kTileHeight;
  const double occupancy =
      static_cast<double>(motionPixels) / static_cast<double>(tilePixels);
  if (occupancy <= 0.25) {
    return ':';
  }
  if (occupancy <= 0.50) {
    return '*';
  }
  if (occupancy <= 0.75) {
    return 'O';
  }
  return '#';
}

void printAsciiHeatmap(const MotionGrid& motionGrid, int camera) {
  std::cout << "Motion Grid — Camera " << camera << "\n\n";
  for (int tileY = 0; tileY < kGridHeight; ++tileY) {
    for (int tileX = 0; tileX < kGridWidth; ++tileX) {
      const int tileIndex =
          camera * kTilesPerCamera + tileY * kGridWidth + tileX;
      std::cout << heatmapSymbol(motionGrid[tileIndex]);
      if (tileX + 1 < kGridWidth) {
        std::cout << ' ';
      }
    }
    std::cout << '\n';
  }
  std::cout << "\nLegend (motion occupancy): . = 0%, : = (0%, 25%], "
               "* = (25%, 50%], O = (50%, 75%], # = (75%, 100%]\n\n";
}

void printHottestTile(const MotionGrid& motionGrid) {
  const auto hottest = std::max_element(motionGrid.begin(), motionGrid.end());
  const int flatIndex = static_cast<int>(hottest - motionGrid.begin());
  const int camera = flatIndex / kTilesPerCamera;
  const int cameraTileIndex = flatIndex % kTilesPerCamera;
  const int tileX = cameraTileIndex % kGridWidth;
  const int tileY = cameraTileIndex / kGridWidth;
  const int xStart = tileX * kTileWidth;
  const int yStart = tileY * kTileHeight;
  const int xEnd = xStart + kTileWidth - 1;
  const int yEnd = yStart + kTileHeight - 1;
  constexpr unsigned int tilePixels = kTileWidth * kTileHeight;
  const double occupancy =
      100.0 * static_cast<double>(*hottest) / static_cast<double>(tilePixels);

  std::cout << "Hottest tile:\n";
  std::cout << "  Camera:               " << camera << '\n';
  std::cout << "  Grid:                 (" << tileX << ", " << tileY << ")\n";
  std::cout << "  Pixel region:         x=" << xStart << ".." << xEnd
            << ", y=" << yStart << ".." << yEnd << '\n';
  std::cout << "  Motion:               " << *hottest << " / " << tilePixels
            << '\n';
  std::cout << "  Occupancy:            " << std::setprecision(1) << occupancy
            << "%\n\n" << std::setprecision(3);
}

bool writePpmHeatmap(
    const MotionGrid& motionGrid,
    int camera,
    const std::string& path
) {
  const int imageWidth = kGridWidth * kPpmTileSize;
  const int imageHeight = kGridHeight * kPpmTileSize;
  constexpr unsigned int tilePixels = kTileWidth * kTileHeight;

  std::ofstream output(path, std::ios::binary);
  if (!output) {
    std::cerr << "Could not create " << path << '\n';
    return false;
  }

  output << "P6\n" << imageWidth << ' ' << imageHeight << "\n255\n";
  for (int imageY = 0; imageY < imageHeight; ++imageY) {
    for (int imageX = 0; imageX < imageWidth; ++imageX) {
      const int tileX = imageX / kPpmTileSize;
      const int tileY = imageY / kPpmTileSize;
      const int tileIndex =
          camera * kTilesPerCamera + tileY * kGridWidth + tileX;

      const bool gridLine =
          imageX % kPpmTileSize == 0 || imageY % kPpmTileSize == 0;
      uint8_t intensity = 24;
      if (!gridLine) {
        const unsigned int count = motionGrid[tileIndex];
        if (count == 0) {
          intensity = 8;
        } else {
          const double occupancy =
              static_cast<double>(count) / static_cast<double>(tilePixels);
          intensity = static_cast<uint8_t>(64.0 + 191.0 * occupancy);
        }
      }

      const char pixel[3] = {
          static_cast<char>(intensity),
          static_cast<char>(intensity),
          static_cast<char>(intensity)
      };
      output.write(pixel, sizeof(pixel));
    }
  }

  if (!output) {
    std::cerr << "Failed while writing " << path << '\n';
    return false;
  }

  return true;
}

int main() {
  const size_t rgbBytes =
      static_cast<size_t>(kTotalPixels) * static_cast<size_t>(kChannels);
  const size_t totalInputBytes = rgbBytes * 2;
  const size_t outputBytes = sizeof(unsigned int) * kTotalTileCounts;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Frame Factory — Motion Grid\n\n";
  std::cout << "Cameras: " << kCameraCount << '\n';
  std::cout << "Resolution: " << kWidth << " x " << kHeight << " RGB\n";
  std::cout << "Grid: " << kGridWidth << " x " << kGridHeight << '\n';
  std::cout << "Tile size: " << kTileWidth << " x " << kTileHeight
            << " pixels\n";
  std::cout << "Tiles per camera: " << kTilesPerCamera << '\n';
  std::cout << "Motion threshold: " << kMotionThreshold
            << " grayscale levels\n";
  std::cout << "Pixels per iteration: " << kTotalPixels << '\n';
  std::cout << "RGB input per iteration: " << totalInputBytes << " bytes\n";
  std::cout << "GPU result copied back: " << outputBytes
            << " bytes (" << kTotalTileCounts << " counters)\n";
  std::cout << "Warm-up iterations: " << kWarmupIterations << '\n';
  std::cout << "Benchmark iterations: " << kBenchmarkIterations << "\n\n";

  std::cout << "Timing scope:\n"
            << "  Frame generation and allocation are outside timed regions.\n"
            << "  CPU time includes counter clearing and the full motion grid.\n"
            << "  GPU total includes H2D, counter clearing, kernel, D2H, and "
               "synchronization.\n"
            << "  FPS means complete 3-camera batches per second.\n\n";

  std::vector<uint8_t> pageablePreviousRgb(rgbBytes);
  std::vector<uint8_t> pageableCurrentRgb(rgbBytes);
  fillFakeMotionFrames(pageablePreviousRgb.data(), pageableCurrentRgb.data());

  const CpuResult cpuResult = runCpuBaseline(
      pageablePreviousRgb.data(),
      pageableCurrentRgb.data()
  );
  printCpuResult(cpuResult);

  int cudaDeviceCount = 0;
  const cudaError_t deviceStatus = cudaGetDeviceCount(&cudaDeviceCount);
  if (deviceStatus != cudaSuccess) {
    std::cerr << "CUDA device query failed: "
              << cudaGetErrorString(deviceStatus) << '\n';
    return EXIT_FAILURE;
  }
  if (cudaDeviceCount == 0) {
    std::cerr << "CUDA device query succeeded, but no CUDA device was found.\n";
    return EXIT_FAILURE;
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

  const GpuResult pageableResult = runGpuBenchmark(
      "GPU pageable",
      pageablePreviousRgb.data(),
      pageableCurrentRgb.data(),
      rgbBytes,
      false,
      cpuResult.averageMs
  );
  const GpuResult pinnedResult = runGpuBenchmark(
      "GPU pinned",
      pinnedPreviousRgb,
      pinnedCurrentRgb,
      rgbBytes,
      true,
      cpuResult.averageMs
  );

  CUDA_CHECK(cudaFreeHost(pinnedPreviousRgb));
  CUDA_CHECK(cudaFreeHost(pinnedCurrentRgb));

  printGpuResult(pageableResult);
  printGpuResult(pinnedResult);

  const bool pageableMatches = validateMotionGrid(
      cpuResult.motionGrid,
      pageableResult.motionGrid,
      pageableResult.name
  );
  const bool pinnedMatches = validateMotionGrid(
      cpuResult.motionGrid,
      pinnedResult.motionGrid,
      pinnedResult.name
  );
  const bool allMatch = pageableMatches && pinnedMatches;

  std::cout << "Validation:\n";
  std::cout << "  CPU/pageable grids match: "
            << (pageableMatches ? "YES" : "NO") << '\n';
  std::cout << "  CPU/pinned grids match:   "
            << (pinnedMatches ? "YES" : "NO") << '\n';
  std::cout << "  CPU/GPU grids match:      "
            << (allMatch ? "YES" : "NO") << "\n\n";

  if (!allMatch) {
    std::cerr << "FATAL: CPU and CUDA motion grids do not match.\n";
    return EXIT_FAILURE;
  }

  const MotionGrid& visualGrid = pinnedResult.motionGrid;
  printAsciiHeatmap(visualGrid, 0);
  printHottestTile(visualGrid);

  constexpr const char* ppmPath = "motion_heatmap.ppm";
  if (!writePpmHeatmap(visualGrid, 0, ppmPath)) {
    return EXIT_FAILURE;
  }
  std::cout << "Wrote " << ppmPath << " (Camera 0, "
            << kGridWidth * kPpmTileSize << " x "
            << kGridHeight * kPpmTileSize << ").\n";

  return EXIT_SUCCESS;
}
