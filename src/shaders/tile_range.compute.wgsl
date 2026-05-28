const WORKGROUP_SIZE: u32 = 256u;

struct PairKey {
    tile_id: u32,
    depth_bits: u32,
};

struct TotalPairs {
    raw_total_pairs: u32,
    sort_pair_count: u32,
    visible_count: u32,
    overflow: u32,
};

struct TileRangeParams {
    tile_count: u32,
};

struct TileRange {
    start: u32,
    end: u32,
};

@group(0) @binding(0)
var<storage, read> total_pairs: TotalPairs;

@group(0) @binding(1)
var<uniform> params: TileRangeParams;

@group(0) @binding(2)
var<storage, read> pair_keys: array<PairKey>;

@group(0) @binding(3)
var<storage, read_write> tile_ranges: array<TileRange>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let id = global_id.x;
    let n = total_pairs.sort_pair_count;

    if id >= n {
        return;
    }

    let curr_tile = pair_keys[id].tile_id;

    if curr_tile >= params.tile_count {
        return;
    }

    let is_start = id == 0u ||
        pair_keys[id - 1u].tile_id != curr_tile;

    if is_start {
        tile_ranges[curr_tile].start = id;
    }

    let is_end = id == n - 1u ||
        pair_keys[id + 1u].tile_id != curr_tile;

    if is_end {
        tile_ranges[curr_tile].end = id + 1u;
    }
}