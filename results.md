# Frame Factory — Motion Grid Results

Successfully tested on:

- Dell Precision 5560
- NVIDIA RTX A2000 Laptop GPU
- 3 simulated 1920 × 1080 RGB camera streams
- 16 × 9 motion grid with 144 tiles per camera
- 5 warm-up iterations and 60 benchmark iterations

## Results

FPS is measured in complete three-camera batches per second.

| Path | H2D | Kernel (clear + grid) | D2H | CPU time / GPU total | FPS | Speedup vs CPU |
|---|---:|---:|---:|---:|---:|---:|
| CPU baseline | — | — | — | 37.605 ms | 26.592 | — |
| GPU pageable | 8.136 ms | 0.507 ms | 0.019 ms | 8.676 ms | 115.263 | 4.334x |
| GPU pinned | 6.223 ms | 0.473 ms | 0.007 ms | 6.719 ms | 148.831 | 5.597x |

Frame generation and allocation are outside the timed regions. CPU time includes counter clearing and the full motion grid. GPU total includes H2D, counter clearing, kernel execution, D2H, and synchronization.

## Validation

- CPU/pageable grids match: **YES**
- CPU/pinned grids match: **YES**
- CPU/GPU grids match: **YES**

All 432 CPU/GPU tile counters matched exactly.

## Spatial result

The hottest tile was:

- Camera: `0`
- Grid position: `(7, 3)`
- Pixel region: `x=840..959, y=360..479`
- Motion: `14400 / 14400`
- Occupancy: `100.0%`

The program wrote `motion_heatmap.ppm` for Camera 0 at 768 × 432 pixels.

## Takeaway

The pinned-memory GPU pipeline completed the full three-camera batch in 6.719 ms at 148.831 FPS, a 5.597x speedup over the 37.605 ms CPU baseline. Its measured breakdown was 6.223 ms H2D, 0.473 ms for counter clearing plus the motion-grid kernel, and 0.007 ms D2H.

Part 2 could tell us that motion happened. Part 3 can now tell us where in the frame that motion happened.
