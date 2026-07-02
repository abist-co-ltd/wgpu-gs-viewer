struct PrefixCounts {
    n0: u32,
    n1: u32,
    n2: u32,
    _pad: u32,
};

struct TotalPairs {
    raw_total_pairs: u32,
    sort_pair_count: u32,
    visible_count: u32,
    overflow: u32,
};

struct PairCountParams {
    max_pairs: u32,
};

@group(0) @binding(0)
var<storage, read> prefix_counts: PrefixCounts;

@group(0) @binding(1)
var<storage, read> block_sums1: array<u32>;

@group(0) @binding(2)
var<storage, read_write> total_pairs: TotalPairs;

@group(0) @binding(3)
var<uniform> params: PairCountParams;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let n2 = prefix_counts.n2;

    var sum = 0u;
    for (var i = 0u; i < n2; i = i + 1u) {
        sum = sum + block_sums1[i];
    }

    let clamped = min(sum, params.max_pairs);

    total_pairs.raw_total_pairs = sum;
    total_pairs.sort_pair_count = clamped;
    total_pairs.visible_count = prefix_counts.n0;
    total_pairs.overflow = select(0u, 1u, sum > params.max_pairs);
}