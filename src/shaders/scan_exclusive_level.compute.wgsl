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
var<storage, read> input_values: array<u32>;

@group(0) @binding(3)
var<storage, read_write> output_values: array<u32>;

@group(0) @binding(4)
var<storage, read_write> block_sums: array<u32>;

var<workgroup> sdata: array<u32, 256>;

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
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let lid = local_id.x;
    let gid = global_id.x;
    let block_id = workgroup_id.x;
    let n = current_n();

    var x = 0u;
    if gid < n {
        x = input_values[gid];
    }

    sdata[lid] = x;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if offset >= WORKGROUP_SIZE {
            break;
        }

        let idx = (lid + 1u) * (offset << 1u) - 1u;
        if idx < WORKGROUP_SIZE {
            sdata[idx] = sdata[idx] + sdata[idx - offset];
        }

        workgroupBarrier();
        offset = offset << 1u;
    }

    let total = sdata[WORKGROUP_SIZE - 1u];

    if lid == WORKGROUP_SIZE - 1u {
        sdata[WORKGROUP_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    offset = WORKGROUP_SIZE >> 1u;
    loop {
        let idx = (lid + 1u) * (offset << 1u) - 1u;

        if idx < WORKGROUP_SIZE {
            let t = sdata[idx - offset];
            sdata[idx - offset] = sdata[idx];
            sdata[idx] = sdata[idx] + t;
        }

        workgroupBarrier();

        if offset == 1u {
            break;
        }

        offset = offset >> 1u;
    }

    if gid < n {
        output_values[gid] = sdata[lid];
    }

    if lid == WORKGROUP_SIZE - 1u {
        block_sums[block_id] = total;
    }
}