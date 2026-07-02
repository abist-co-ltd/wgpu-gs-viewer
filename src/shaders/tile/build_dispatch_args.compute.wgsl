const WORKGROUP_SIZE: u32 = 256u;

struct DispatchIndirectArgs {
    x: u32,
    y: u32,
    z: u32,
    _pad: u32,
};

struct PrefixCounts {
    n0: u32,
    n1: u32,
    n2: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read_write> visible_count: atomic<u32>;

@group(0) @binding(1)
var<storage, read_write> dispatch_args: array<DispatchIndirectArgs, 5>;

@group(0) @binding(2)
var<storage, read_write> prefix_counts: PrefixCounts;

fn ceil_div_256(x: u32) -> u32 {
    return (x + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
}

fn make_args(group_count: u32) -> DispatchIndirectArgs {
    return DispatchIndirectArgs(
        group_count,
        1u,
        1u,
        0u,
    );
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let n0 = atomicLoad(&visible_count);
    let n1 = ceil_div_256(n0);
    let n2 = ceil_div_256(n1);

    prefix_counts.n0 = n0;
    prefix_counts.n1 = n1;
    prefix_counts.n2 = n2;
    prefix_counts._pad = 0u;

    dispatch_args[0] = make_args(n1);              // scan0
    dispatch_args[1] = make_args(n2);              // scan1
    dispatch_args[2] = make_args(ceil_div_256(n2)); // scan2

    dispatch_args[3] = make_args(n2);              // add1
    dispatch_args[4] = make_args(n1);              // add0
}