const WORKGROUP_SIZE: u32 = 256u;

struct TileRangeParams {
    tile_count: u32,
};

struct TileRange {
    start: u32,
    end: u32,
};

@group(0) @binding(0)
var<uniform> params: TileRangeParams;

@group(0) @binding(1)
var<storage, read_write> tile_ranges: array<TileRange>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tile_id = global_id.x;

    if tile_id >= params.tile_count {
        return;
    }

    tile_ranges[tile_id].start = 0xffffffffu;
    tile_ranges[tile_id].end = 0xffffffffu;
}