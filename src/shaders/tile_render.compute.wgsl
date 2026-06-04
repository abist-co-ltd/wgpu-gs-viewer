const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const WORKGROUP_SIZE: u32 = TILE_W * TILE_H; // 256

struct SceneUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    gaussian_count: u32,
    screen_size: vec2<u32>,
    near_far: vec2<f32>,
    tan_fov: vec2<f32>,
    time: f32,
    _pad0: u32,
};

struct PreprocessOutput {
    conic_opacity: vec4<f32>,   // conic, opacity
    color_radius: vec4<f32>,    // color, radius
    tile_rect: vec4<u32>,
    uv_depth: vec4<f32>, // x, y, depth, pad
};

struct TileRange {
    start: u32,
    end: u32,
};

struct TotalPairs {
    raw_total_pairs: u32,
    sort_pair_count: u32,
    visible_count: u32,
    overflow: u32,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> preprocess_outputs: array<PreprocessOutput>;

@group(0) @binding(2)
var<storage, read> tile_ranges: array<TileRange>;

@group(0) @binding(3)
var<storage, read> sorted_ids: array<u32>;

@group(0) @binding(4)
var output_tex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(5)
var<storage, read> total_pairs: TotalPairs;

var<workgroup> sh_uv: array<vec2<f32>, 256>;
var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_b: array<f32, 256>;
var<workgroup> sh_c: array<f32, 256>;
var<workgroup> sh_opacity: array<f32, 256>;
var<workgroup> sh_color: array<vec3<f32>, 256>;

@compute @workgroup_size(TILE_W, TILE_H, 1)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let width = scene.screen_size.x;
    let height = scene.screen_size.y;

    let tiles_width = (width + TILE_W - 1u) / TILE_W;
    let tile_x = workgroup_id.x;
    let tile_y = workgroup_id.y;
    let tile_id = tile_x + tile_y * tiles_width;

    let px = tile_x * TILE_W + local_id.x;
    let py = tile_y * TILE_H + local_id.y;

    let skip_pixel = (px >= width) || (py >= height);

    let range = tile_ranges[tile_id];
    let start = range.start;
    let end = range.end;

    // empty tile
    if start == 0xffffffffu || end == 0xffffffffu || start >= end {
        if !skip_pixel {
            textureStore(output_tex, vec2<i32>(i32(px), i32(py)), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        }
        return;
    }

    var done = false;
    var T = 1.0;
    var accum = vec3<f32>(0.0, 0.0, 0.0);

    // pixel center
    let curr = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

    for (var batch_start = start; batch_start < end; batch_start = batch_start + WORKGROUP_SIZE) {
        let global_idx = batch_start + local_index;

        if global_idx < end && global_idx < total_pairs.sort_pair_count {
            let visible_idx = sorted_ids[global_idx];

            if visible_idx < total_pairs.visible_count {
                let attr = preprocess_outputs[visible_idx];

                sh_uv[local_index] = attr.uv_depth.xy;
                sh_a[local_index] = attr.conic_opacity.x;
                sh_b[local_index] = attr.conic_opacity.y;
                sh_c[local_index] = attr.conic_opacity.z;
                sh_opacity[local_index] = attr.conic_opacity.w;
                sh_color[local_index] = attr.color_radius.xyz;
            } else {
                sh_opacity[local_index] = 0.0;
            }
        } else {
            sh_opacity[local_index] = 0.0;
        }

        workgroupBarrier();

        if !done {
            if skip_pixel {
                done = true;
            } else {
                let batch_size = min(WORKGROUP_SIZE, end - batch_start);

                for (var j = 0u; j < batch_size; j = j + 1u) {
                    let op = sh_opacity[j];
                    if op == 0.0 {
                        continue;
                    }

                    let d = sh_uv[j] - curr;
                    let dx = d.x;
                    let dy = d.y;

                    let quad = sh_a[j] * dx * dx +
                        sh_c[j] * dy * dy +
                        2.0 * sh_b[j] * dx * dy;

                    let power = -0.5 * quad;

                    if power > 0.0 {
                        continue;
                    }

                    var alpha = op * exp(power);
                    alpha = min(alpha, 0.99);

                    if alpha < (1.0 / 255.0) {
                        continue;
                    }

                    accum += sh_color[j] * (alpha * T);
                    T *= (1.0 - alpha);

                    if T < 1e-2 {
                        done = true;
                        break;
                    }
                }
            }
        }
        workgroupBarrier();
    }

    if !skip_pixel {
        textureStore(output_tex, vec2<i32>(i32(px), i32(py)), vec4<f32>(accum, 1.0));
    }
}