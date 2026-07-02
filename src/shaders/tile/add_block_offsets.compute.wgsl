const WORKGROUP_SIZE: u32 = 256u;

struct PrefixLevelParams {
    level: u32,
};

struct PrefixCounts {
    n0: u32,
    n1: u32,
    n2: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<uniform> params: PrefixLevelParams;

@group(0) @binding(1)
var<storage, read> prefix_counts: PrefixCounts;

@group(0) @binding(2)
var<storage, read_write> values: array<u32>;

@group(0) @binding(3)
var<storage, read> block_offsets: array<u32>;

fn current_n() -> u32 {
    if params.level == 0u {
        return prefix_counts.n0;
    }
    if params.level == 1u {
        return prefix_counts.n1;
    }
    return prefix_counts.n2;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    let n = current_n();

    if gid >= n {
        return;
    }

    let block_id = gid >> 8u;
    let add_value = block_offsets[block_id];

    values[gid] = values[gid] + add_value;
}