# Frame Factory — Motion Grid

Something moved. Where did it move?

This small CUDA/C++ benchmark is Week 11, Part 3 of the Frame Factory learning series. It turns deterministic RGB frame differences into a spatial motion map for three simulated cameras.

## Series progression

- [Part 1 — Frame Factory](https://github.com/KhanAymann/frame-factory): move camera-like frames from CPU to GPU, run a simple CUDA kernel, and return the result quickly.
- [Part 2 — Motion Map](https://github.com/KhanAymann/frame-factory-motion-map): compare consecutive frames and count how many pixels changed.
- Part 3 — Motion Grid: divide each frame into spatial bins and count where those changes occurred.

```text
previous frame + current frame
             ↓
      pixel difference
             ↓
     binary motion mask
             ↓
    16 × 9 spatial bins
             ↓
       motion heatmap
```

## What it does

The program simulates three deterministic 1920 × 1080 RGB camera streams. A bright rectangle moves between the previous and current frame and crosses several grid cells. It uses no OpenCV, machine-learning model, or external asset.

Each frame is divided into a 16 × 9 grid. Because the source is exactly 1920 × 1080, every tile is 120 × 120 pixels. The output is 144 motion counts per camera, or 432 counts for the complete three-camera batch.

It benchmarks three implementations:

- CPU baseline
- CUDA with pageable host memory
- CUDA with pinned host memory allocated by `cudaMallocHost`

The CUDA kernel is intentionally straightforward: one thread compares one pixel, converts both RGB values to grayscale, applies the threshold, finds the pixel's camera and tile, then uses `atomicAdd` to increment that tile's counter.

After the benchmark, the program compares every CUDA tile count with the CPU result. A mismatch reports the memory mode, camera, tile coordinates, CPU count, and GPU count, then exits with failure.

After successful validation, it prints an ASCII heatmap and the hottest tile from the pinned-memory CUDA result, and writes the same Camera 0 grid to `motion_heatmap.ppm`. The PPM needs no image library; many Linux image viewers open it directly.

## Why this matters

A global motion count can tell a robot that something moved. A spatial map begins to tell it where activity occurred.

This is still not object detection, tracking, segmentation, or scene understanding. It is a deliberately small perception step: frame difference → binary motion → spatial motion map.

## Benchmark scope

Frame generation and memory allocation happen outside the timed regions. The CPU and CUDA paths both perform the same grayscale difference, threshold, and 16 × 9 binning across all three cameras.

- CPU time includes clearing all 432 counters and computing the full motion grid.
- H2D is the CUDA-event time for both previous/current RGB transfers.
- Kernel includes clearing the 432 device counters and running the motion-grid kernel.
- D2H is the CUDA-event time for returning all 432 counters.
- GPU total is host wall time for H2D, counter clearing, kernel, D2H, and the required stream synchronization.
- FPS means complete three-camera batches per second, not an aggregate per-camera rate.

The per-stage CUDA-event values may not sum exactly to host wall time because the total also includes host API, pageable-memory staging, and synchronization overhead.

Five warm-up iterations run before 60 measured iterations. Synthetic frame generation is deterministic and is not part of those timings.

The CPU loop uses a compiler barrier after every completed grid so an optimizing build cannot collapse repeated identical benchmark iterations.

## Build and run

This project targets an NVIDIA CUDA machine. Build it on the Dell Precision 5560:

```bash
nvcc -O3 -std=c++17 frame_factory_motion_grid.cu -o frame_factory_motion_grid
```

Run it:

```bash
./frame_factory_motion_grid
```

Save the benchmark output:

```bash
./frame_factory_motion_grid | tee motion_results.txt
```

The runtime creates `motion_heatmap.ppm` in the current directory.

## Results status

Part 3 benchmark results are intentionally pending execution on the Dell Precision 5560 with its NVIDIA RTX A2000 Laptop GPU. The source was prepared on an Apple-silicon Mac, which cannot build or run CUDA. No CUDA build result or performance number is claimed from the Mac.

Add `results.md` only after collecting real output on the NVIDIA machine. Do not treat inherited Part 2 measurements as Part 3 results: this version returns 432 counters and uses per-tile atomic accumulation, so it is a different workload.

## Files

- `frame_factory_motion_grid.cu` — CPU/CUDA benchmark and visualization source
- `motion_heatmap.ppm` — generated Camera 0 heatmap
- `motion_results.txt` — optional captured terminal output after a real run
- `results.md` — optional summary after real measurements exist
