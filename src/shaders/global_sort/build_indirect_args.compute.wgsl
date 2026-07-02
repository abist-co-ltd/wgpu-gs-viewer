const WORKGROUP_SIZE: u32 = 256u;
const MAX_RADIX_WORKGROUPS: u32 = 256u;
const RADIX_SORT_PASSES: u32 = 4u;
const RADIX_BITS_PER_PASS: u32 = 8u;

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
};

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(0)
var<storage, read_write> visible_count: atomic<u32>;

@group(0) @binding(1)
var<storage, read_write> radix_dispatch_args: DispatchIndirectArgs;

@group(0) @binding(2)
var<storage, read_write> radix_params: array<RadixSortParams, 4>;

@group(0) @binding(3)
var<storage, read_write> draw_args: DrawIndirectArgs;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let count = atomicLoad(&visible_count);

    // Radix sort passes
    let total_blocks = (count + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    let radix_workgroups = min(total_blocks, MAX_RADIX_WORKGROUPS);

    var blocks_per_workgroup = 0u;

    if radix_workgroups > 0u {
        blocks_per_workgroup = (total_blocks + radix_workgroups - 1u) / radix_workgroups;
    }

    radix_dispatch_args.x = radix_workgroups;
    radix_dispatch_args.y = 1u;
    radix_dispatch_args.z = 1u;

    for (var i = 0u; i < RADIX_SORT_PASSES; i += 1u) {
        radix_params[i].num_elements = count;
        radix_params[i].shift = i * RADIX_BITS_PER_PASS;
        radix_params[i].num_workgroups = radix_workgroups;
        radix_params[i].num_blocks_per_workgroup = blocks_per_workgroup;
    }

    // render pass
    draw_args.vertex_count = 4u;
    draw_args.instance_count = count;
    draw_args.first_vertex = 0u;
    draw_args.first_instance = 0u;
}