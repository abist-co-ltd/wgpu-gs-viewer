const WORKGROUP_SIZE: u32 = 256u;
const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const SIGMA_SCALE: f32 = 3.0;

struct PreprocessOutput {
    conic_opacity: vec4<f32>,
    color_radius: vec4<f32>,
    tile_rect: vec4<u32>,
    uv_depth: vec4<f32>, // x, y, depth, pad
};

struct DuplicateParams {
    tiles_width: u32,
    tiles_height: u32,
    max_pairs: u32,
    _pad: u32,
};

struct PairKey {
    tile_id: u32,
    depth_bits: u32,
};

@group(0) @binding(0)
var<uniform> params: DuplicateParams;

@group(0) @binding(1)
var<storage, read> outputs: array<PreprocessOutput>;

@group(0) @binding(2)
var<storage, read> offsets: array<u32>;

@group(0) @binding(3)
var<storage, read_write> pair_keys: array<PairKey>;

@group(0) @binding(4)
var<storage, read_write> pair_values: array<u32>;

@group(0) @binding(5)
var<storage, read_write> visible_count: atomic<u32>;

fn tile_may_intersect_conic(
    uv: vec2<f32>,
    conic: vec3<f32>,
    tile_x: u32,
    tile_y: u32,
) -> bool {
    let tile_min = vec2<f32>(
        f32(tile_x * TILE_W),
        f32(tile_y * TILE_H),
    );

    let tile_max = tile_min + vec2<f32>(
        f32(TILE_W),
        f32(TILE_H),
    );

    let tile_center = 0.5 * (tile_min + tile_max);
    let d = tile_center - uv;

    let a = conic.x;
    let b = conic.y;
    let c = conic.z;

    let q_center = a * d.x * d.x +
        2.0 * b * d.x * d.y +
        c * d.y * d.y;

    let md = vec2<f32>(
        a * d.x + b * d.y,
        b * d.x + c * d.y,
    );

    let half_diag = 0.5 * sqrt(f32(TILE_W * TILE_W + TILE_H * TILE_H));
    let lower_bound = q_center - 2.0 * length(md) * half_diag;

    return lower_bound <= SIGMA_SCALE * SIGMA_SCALE;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let count = atomicLoad(&visible_count);
    if idx >= count {
        return;
    }

    let attr = outputs[idx];

    let tile_rect = attr.tile_rect;

    let min_tx = tile_rect.x;
    let min_ty = tile_rect.y;
    let max_tx = tile_rect.z;
    let max_ty = tile_rect.w;

    // invalid / empty
    if max_tx <= min_tx || max_ty <= min_ty {
        return;
    }

    let depth = attr.uv_depth.z;
    let depth_bits = bitcast<u32>(depth);

    let base = offsets[idx];

    let uv = attr.uv_depth.xy;
    let conic = attr.conic_opacity.xyz;

    var write = 0u;

    for (var ty = min_ty; ty < max_ty; ty = ty + 1u) {
        let row_base = ty * params.tiles_width;

        for (var tx = min_tx; tx < max_tx; tx = tx + 1u) {
            // IMPORTANT: 
            //  Emit pairs only for tiles that pass the same conservative intersection test
            //  used in preprocess.compute.wgsl. The number of emitted pairs must match
            //  tiles_touched[visible_idx], otherwise offsets become invalid.
            if !tile_may_intersect_conic(uv, conic, tx, ty) {
                continue;
            }

            let tile_id = row_base + tx;
            let out_index = base + write;

            if out_index >= params.max_pairs {
                return;
            }

            pair_keys[out_index] = PairKey(
                tile_id,
                depth_bits,
            );

            pair_values[out_index] = idx;

            write = write + 1u;
        }
    }
}