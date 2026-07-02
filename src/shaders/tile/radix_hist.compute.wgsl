/**
 * Radix sort implementation for wgpu/WGSL.
 *
 * This shader is ported and adapted from VkRadixSort:
 *   https://github.com/MircoWerner/VkRadixSort
 *
 * VkRadixSort was written by Mirco Werner and is based on Intel Embree's GPU sort:
 *   https://github.com/embree/embree/blob/v4.0.0-ploc/kernels/rthwif/builder/gpu/sort.h
 *
 * Major changes in this version:
 * - Rewritten from Vulkan GLSL to WGSL for wgpu/WebGPU.
 * - Replaced push constants and specialization constants with uniform/storage buffers
 *   and fixed WGSL constants.
 * - Removed Vulkan subgroup assumptions so the shader can run on WebGPU.
 * - Replaced the original single u32/u64-style key input with split pair keys:
 *
 *     PairKey {
 *         tile_id: u32,
 *         depth_bits: u32,
 *     }
 *
 *   Conceptually this corresponds to a 64-bit key layout:
 *
 *     upper 32 bits = tile_id
 *     lower 32 bits = depth_bits
 *
 * - Added pair-value sorting support in the scatter pass so Gaussian indices move
 *   together with their keys.
 * - Uses a transposed histogram layout:
 *
 *     histograms[bin * num_workgroups + workgroup]
 *
 *   so each bin's workgroup counts can be scanned/read contiguously.
 * - Designed for 3D Gaussian Splatting tile/depth pair sorting.
 *
 * Original license:
 *
 * MIT License
 *
 * Copyright (c) 2023 Mirco Werner
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SORT_BINS: u32 = 256u;

struct PairKey {
    tile_id: u32,
    depth_bits: u32,
};

struct RadixSortParams {
    num_elements: u32,
    shift: u32,
    num_workgroups: u32,
    num_blocks_per_workgroup: u32,
};

struct RadixPassIndex {
    value: u32,
};

@group(0) @binding(0)
var<uniform> pass_index: RadixPassIndex;

fn get_params() -> RadixSortParams {
    return radix_params[pass_index.value];
}

@group(0) @binding(1)
var<storage, read> radix_params: array<RadixSortParams, 8>;

@group(0) @binding(2)
var<storage, read> keys_in: array<PairKey>;

@group(0) @binding(3)
var<storage, read_write> histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 256>;

fn key_digit(key: PairKey, shift: u32) -> u32 {
    if shift < 32u {
        return (key.depth_bits >> shift) & 255u;
    }

    return (key.tile_id >> (shift - 32u)) & 255u;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lid = local_id.x;
    let wid = workgroup_id.x;
    let params = get_params();

    // Initialize one bin per thread.
    atomicStore(&histogram[lid], 0u);
    workgroupBarrier();

    for (var block = 0u; block < params.num_blocks_per_workgroup; block = block + 1u) {
        let element_id = wid * params.num_blocks_per_workgroup * WORKGROUP_SIZE +
            block * WORKGROUP_SIZE +
            lid;

        if element_id < params.num_elements {
            let key = keys_in[element_id];
            let bin = key_digit(key, params.shift);
            atomicAdd(&histogram[bin], 1u);
        }
    }

    workgroupBarrier();

    if wid < params.num_workgroups {
        histograms[lid * params.num_workgroups + wid] = atomicLoad(&histogram[lid]);
    }
}