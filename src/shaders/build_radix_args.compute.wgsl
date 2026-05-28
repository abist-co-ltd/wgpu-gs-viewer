const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SORT_PASSES: u32 = 8u;
const MAX_RADIX_WORKGROUPS: u32 = 256u;

struct TotalPairs {
    raw_total_pairs: u32,
    sort_pair_count: u32,
    visible_count: u32,
    overflow: u32,
};

struct RadixSortParams {
    num_elements: u32,
    shift: u32,
    num_workgroups: u32,
    num_blocks_per_workgroup: u32,
};

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> total_pairs: TotalPairs;

@group(0) @binding(1)
var<storage, read_write> radix_params: array<RadixSortParams, 8>;

@group(0) @binding(2)
var<storage, read_write> radix_dispatch_args: DispatchIndirectArgs;

@group(0) @binding(3)
var<storage, read_write> tile_range_dispatch_args: DispatchIndirectArgs;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let n = total_pairs.sort_pair_count;

    let num_blocks = max(1u, (n + WORKGROUP_SIZE * MAX_RADIX_WORKGROUPS - 1u) / (WORKGROUP_SIZE * MAX_RADIX_WORKGROUPS));
    let groups = (n + WORKGROUP_SIZE * num_blocks - 1u) / (WORKGROUP_SIZE * num_blocks);

    radix_dispatch_args.x = groups;
    radix_dispatch_args.y = 1u;
    radix_dispatch_args.z = 1u;
    radix_dispatch_args._pad = 0u;

    for (var i = 0u; i < RADIX_SORT_PASSES; i = i + 1u) {
        radix_params[i].num_elements = n;
        radix_params[i].shift = i * 8u;
        radix_params[i].num_workgroups = groups;
        radix_params[i].num_blocks_per_workgroup = num_blocks;
    }

    tile_range_dispatch_args.x = (n + 255u) / 256u;
    tile_range_dispatch_args.y = 1u;
    tile_range_dispatch_args.z = 1u;
    tile_range_dispatch_args._pad = 0u;
}