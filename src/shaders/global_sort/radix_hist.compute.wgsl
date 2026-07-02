const WORKGROUP_SIZE: u32 = 256u;
const RADIX_SORT_BINS: u32 = 256u;
const RADIX_SORT_PASSES: u32 = 4u;

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

@group(0) @binding(1)
var<storage, read> radix_params: array<RadixSortParams, 4>;

@group(0) @binding(2)
var<storage, read> keys_in: array<u32>;

@group(0) @binding(3)
var<storage, read_write> histograms: array<u32>;

var<workgroup> histogram: array<atomic<u32>, 256>;

fn get_params() -> RadixSortParams {
    return radix_params[pass_index.value];
}

fn key_digit(
    key: u32,
    shift: u32,
) -> u32 {
    return (key >> shift) & 255u;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id)
    local_id: vec3<u32>,
    @builtin(workgroup_id)
    workgroup_id: vec3<u32>,
) {
    let lid = local_id.x;
    let wid = workgroup_id.x;
    let params = get_params();

    atomicStore(&histogram[lid], 0u);

    workgroupBarrier();

    for (var block = 0u; block < params.num_blocks_per_workgroup; block = block + 1u) {
        let element_id = wid * params.num_blocks_per_workgroup * WORKGROUP_SIZE + block * WORKGROUP_SIZE + lid;
        if element_id < params.num_elements {
            let key = keys_in[element_id];

            let bin = key_digit(key, params.shift);
            atomicAdd(
                &histogram[bin],
                1u,
            );
        }
    }

    workgroupBarrier();

    histograms[lid * params.num_workgroups + wid] = atomicLoad(&histogram[lid]);
}