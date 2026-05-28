/**
 * Part of the wgpu/WGSL radix sort port adapted from https://github.com/MircoWerner/VkRadixSort.
 * See radix_hist.compute.wgsl for full attribution and MIT license text.
 */

const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SORT_BINS: u32 = 256u;
const FLAG_WORDS: u32 = 8u; // WORKGROUP_SIZE / 32

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

struct BinFlags {
    words: array<atomic<u32>, 8>,
};

@group(0) @binding(1)
var<storage, read> radix_params: array<RadixSortParams, 8>;

@group(0) @binding(2)
var<storage, read> keys_in: array<PairKey>;

@group(0) @binding(3)
var<storage, read_write> keys_out: array<PairKey>;

@group(0) @binding(4)
var<storage, read> values_in: array<u32>;

@group(0) @binding(5)
var<storage, read_write> values_out: array<u32>;

@group(0) @binding(6)
var<storage, read> histograms: array<u32>;

var<workgroup> global_offsets: array<u32, 256>;
var<workgroup> bin_scan: array<u32, 256>;
var<workgroup> bin_flags: array<BinFlags, 256>;

fn key_digit(key: PairKey, shift: u32) -> u32 {
    if shift < 32u {
        return (key.depth_bits >> shift) & 255u;
    }
    return (key.tile_id >> (shift - 32u)) & 255u;
}

fn count_bits_before(bin: u32, word_id: u32, bit_mask: u32) -> u32 {
    var prefix = 0u;
    for (var i = 0u; i < FLAG_WORDS; i = i + 1u) {
        let bits = atomicLoad(&bin_flags[bin].words[i]);
        if i < word_id {
            prefix = prefix + countOneBits(bits);
        }
        if i == word_id {
            prefix = prefix + countOneBits(bits & (bit_mask - 1u));
        }
    }
    return prefix;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lid = local_id.x;
    let wid = workgroup_id.x;
    let params = get_params();

    // TODO:
    //  Scatter pass currently recomputes global_offsets from histograms in each workgroup.
    //  This avoids an extra pass, but may be costly when num_workgroups is large.
    //  Benchmark an alternative that precomputes bin_workgroup_offsets[bin][wg]
    //  in a separate pass, so scatter can just read the offset.

    // Sum bin counts across all workgroups
    var total_count = 0u;
    var local_offset = 0u;
    let base = lid * params.num_workgroups;
    for (var j = 0u; j < params.num_workgroups; j = j + 1u) {
        if j == wid {
            local_offset = total_count;
        }
        total_count = total_count + histograms[base + j];
    }

    bin_scan[lid] = total_count;
    workgroupBarrier();

    // Blelloch exclusive prefix scan
    var scan_offset = 1u;
    loop {
        if scan_offset >= RADIX_SORT_BINS { break; }
        let idx = (lid + 1u) * (scan_offset << 1u) - 1u;
        if idx < RADIX_SORT_BINS {
            bin_scan[idx] = bin_scan[idx] + bin_scan[idx - scan_offset];
        }
        workgroupBarrier();
        scan_offset = scan_offset << 1u;
    }
    if lid == RADIX_SORT_BINS - 1u {
        bin_scan[RADIX_SORT_BINS - 1u] = 0u;
    }
    workgroupBarrier();
    scan_offset = RADIX_SORT_BINS >> 1u;
    loop {
        let idx = (lid + 1u) * (scan_offset << 1u) - 1u;
        if idx < RADIX_SORT_BINS {
            let t = bin_scan[idx - scan_offset];
            bin_scan[idx - scan_offset] = bin_scan[idx];
            bin_scan[idx] = bin_scan[idx] + t;
        }
        workgroupBarrier();
        if scan_offset == 1u { break; }
        scan_offset = scan_offset >> 1u;
    }
    global_offsets[lid] = bin_scan[lid] + local_offset;
    workgroupBarrier();

    // scatter each block of WORKGROUP_SIZE elements.
    for (var block = 0u; block < params.num_blocks_per_workgroup; block = block + 1u) {
        // clear bin flags
        for (var word = 0u; word < FLAG_WORDS; word = word + 1u) {
            atomicStore(&bin_flags[lid].words[word], 0u);
        }
        workgroupBarrier();

        let element_id = wid * params.num_blocks_per_workgroup * WORKGROUP_SIZE
            + block * WORKGROUP_SIZE + lid;

        var key = PairKey(0u, 0u);
        var value = 0u;
        var bin = 0u;
        let valid = element_id < params.num_elements;

        if valid {
            key = keys_in[element_id];
            value = values_in[element_id];
            bin = key_digit(key, params.shift);
            let word_id = lid / 32u;
            let bit = 1u << (lid & 31u);
            atomicOr(&bin_flags[bin].words[word_id], bit);
        }
        workgroupBarrier();

        var local_prefix = 0u;
        var bin_count_in_block = 0u;

        if valid {
            let word_id = lid / 32u;
            let bit = 1u << (lid & 31u);
            local_prefix = count_bits_before(bin, word_id, bit);

            for (var i = 0u; i < FLAG_WORDS; i = i + 1u) {
                bin_count_in_block = bin_count_in_block + countOneBits(atomicLoad(&bin_flags[bin].words[i]));
            }

            keys_out[global_offsets[bin] + local_prefix] = key;
            values_out[global_offsets[bin] + local_prefix] = value;
        }

        workgroupBarrier();

        if valid {
            // The last element in this bin advances the shared offset for the next block.
            if local_prefix == bin_count_in_block - 1u {
                global_offsets[bin] = global_offsets[bin] + bin_count_in_block;
            }
        }
        workgroupBarrier();
    }
}