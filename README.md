# Frame Factory Motion Map

A small CUDA/C++ benchmark for an early robotics perception step:

previous camera frame + current camera frame -> motion count

Before a robot can fold laundry, it has to see.

Before it understands what it sees, it has to move camera data through memory and turn pixels into useful signals.

This project simulates three 1080p RGB camera streams and benchmarks how quickly a CPU and GPU can detect motion between previous and current frames.

## What It Simulates

- 3 camera streams
- 1920 x 1080 RGB frames
- previous-frame and current-frame buffers
- grayscale difference
- thresholding into motion / no motion
- motion pixel count

Each benchmark step processes:

- 6,220,800 pixels
- 37.325 MB of RGB input data
- one 4-byte motion count copied back from the GPU

## What It Measures

The benchmark compares:

- CPU baseline
- GPU with pageable CPU memory
- GPU with pinned CPU memory using cudaMallocHost

It reports:

- CPU time
- CPU FPS
- CPU -> GPU copy time
- CUDA kernel time
- GPU -> CPU copy time
- total GPU pipeline time
- GPU FPS
- speedup vs CPU
- pinned vs pageable speedup
- whether GPU and CPU motion counts match

## Results

Tested on a Dell Precision 5560 with an NVIDIA RTX A2000 Laptop GPU.

| Pipeline            |      Time |         FPS | Notes                                               |
| ------------------- | --------: | ----------: | --------------------------------------------------- |
| CPU baseline        | 10.489 ms |  95.340 FPS | CPU implementation                                  |
| GPU pageable memory | 17.690 ms |  56.531 FPS | Slower than CPU because transfer overhead dominated |
| GPU pinned memory   |  7.144 ms | 139.976 FPS | Faster than CPU after reducing transfer cost        |

Motion pixel count: 83,520

Pinned memory vs pageable memory: 2.476x faster

## Main Takeaway

The CUDA kernel itself was fast: around 0.416 ms.

The bottleneck was memory transfer.

Using pinned CPU memory cut the CPU -> GPU transfer time enough for the GPU pipeline to beat the CPU baseline.

That is the useful robotics lesson: real-time perception is not just about smarter models. It also depends on latency, throughput, memory movement, and synchronization.

## Build and Run

Compile:

nvcc -O3 -std=c++17 frame_factory_motion.cu -o frame_factory_motion

Run:

./frame_factory_motion

Save output:

./frame_factory_motion | tee motion_results.txt

## Files

- frame_factory_motion.cu — CUDA/C++ benchmark source
- motion_results.txt — saved benchmark output
- results.md — summarized benchmark results

## Hardware

Tested on:

- Dell Precision 5560
- NVIDIA RTX A2000 Laptop GPU
- Linux Mint
- CUDA / nvcc

## Scope

This is not a full robotics system.

It does not use a real camera, ROS, SLAM, object detection, segmentation, tracking, or robot control.

It is a small CUDA benchmark focused on one early perception step:

How quickly can a system compare camera frames and produce a basic motion signal?
