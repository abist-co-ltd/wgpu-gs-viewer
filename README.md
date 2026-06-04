# Gaussian Splatting viewer

<table>
  <tr>
    <td>
      <img src="./docs/demo-train-3dgs.gif" width="360" alt="3DGS demo">
    </td>
    <td>
      <img src="./docs/demo-flame-stg-lite.gif" width="360" alt="STG Lite demo">
    </td>
  </tr>
</table>

A Gaussian Splatting viewer implemented with Rust and wgpu.

It loads trained 3DGS and Space-Time Gaussian Lite PLY files and renders them using a GPU-based tile rendering pipeline.

## Main features

* Load 3DGS `.ply` files
* Load Space-Time Gaussian Lite `.ply` files
* Render Gaussian splats with wgpu
* GPU-based tile rendering pipeline
* WebGPU support
* Native desktop support

## Rendering pipeline

This viewer renders Gaussian Splatting scenes using a GPU-based tile rendering pipeline.

The overall pipeline is shared by both 3DGS and Space-Time Gaussian Lite scenes.  
The main difference is the **Preprocess pass**.

### Preprocess pass

For 3DGS scenes, each Gaussian is processed as a static 3D Gaussian.

- Projects each Gaussian to screen space
- Evaluates SH degree 3 color
- Computes the 2D covariance/conic, opacity, depth, radius, and touched tile bounds

For Space-Time Gaussian Lite scenes, each Gaussian is first evaluated at the current time `t`.

- Evaluates `position(t)`, `rotation(t)`, and `opacity(t)`
- Projects the evaluated Gaussian to screen space
- Uses base color only
- Computes the 2D covariance/conic, depth, radius, and touched tile bounds

### Shared GPU passes

After the preprocess pass, both formats use the same GPU pipeline.

- **Prefix scan pass**: computes offsets from the number of tiles touched by each visible Gaussian.
- **Duplicate pass**: expands each visible Gaussian into per-tile entries with tile ID and depth.
- **Radix sort pass**: sorts duplicated entries by tile ID and depth.
- **Tile range pass**: finds the range of sorted entries belonging to each tile.
- **Render pass**: blends sorted Gaussians per tile.

## Controls

- Drag and drop a `.ply` file to load a Gaussian Splatting scene
- W / A / S / D: rotate camera
- Q / E: zoom in / out

## Build Locally

### Native Desktop

```
cargo run --release
```

### Web

Build the WebAssembly package:

```
wasm-pack build --target web --release
```

Then serve the project directory with a local HTTP server.

For example:

```
python3 -m http.server 8080
```

Open the following URL in your browser:

```
http://localhost:8080
```
## Version history

### v-0.2.0

Added Space-Time Gaussian Lite viewer support.

- Added Space-Time Gaussian Lite PLY loading
- Added time-dependent preprocess for `position(t)`, `rotation(t)`, and `opacity(t)`
- Added 4DGS playback support

### v-0.1.0

Initial 3D Gaussian Splatting viewer.

- Added 3DGS PLY loading
- Added GPU-based tile rendering pipeline
- Added WebGPU and native desktop support

## Notes

* Tested on macOS with Apple M4 and 24 GB RAM.
* Web version tested on Chrome 148 and Safari 26.
* Performance and compatibility may vary depending on GPU, browser, and WebGPU implementation.
* 4DGS support currently means STG-Lite support. Other 4DGS variants are not supported yet.

## Dataset Attribution

[The 3DGS demo GIF](docs/demo-train-3dgs.gif) was generated using a trained PLY file from the Gaussian Splatting dataset by Paula Ramos.

Dataset: https://huggingface.co/datasets/Voxel51/gaussian_splatting
License: Apache License 2.0

The dataset was created using the official 3D Gaussian Splatting method:

Kerbl et al., “3D Gaussian Splatting for Real-Time Radiance Field Rendering”, 2023.

[The STG-Lite demo GIF](docs/demo-flame-stg-lite.gif) was generated using a pretrained PLY model from the official SpacetimeGaussians project.

* Project page: https://oppo-us-research.github.io/SpacetimeGaussians-website/
* Official implementation: https://github.com/oppo-us-research/SpacetimeGaussians

Please refer to the original repositories and datasets for their licenses and additional use limitations.

## Acknowledgements

STG-Lite support is based on the representation introduced in **Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis**.

The STG-Lite implementation was developed with reference to [splatv](https://github.com/antimatter15/splatv) and the official [SpacetimeGaussians](https://github.com/oppo-us-research/SpacetimeGaussians) implementation.

The radix sort implementation is based on [VkRadixSort](https://github.com/MircoWerner/VkRadixSort) and adapted for wgpu/WGSL.
