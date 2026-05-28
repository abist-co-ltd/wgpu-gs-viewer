
const TILE_W: u32 = 16u;
const TILE_H: u32 = 16u;
const MAX_RADIUS: f32 = 1024.0;
const SIGMA_SCALE: f32 = 3.0;

// SH constants (standard real SH basis scaling constants for l<=3)
const SH0: f32 = 0.28209479177387814;
const SH1: f32 = 0.4886025119029199;

const SH2_0: f32 = 1.0925484305920792;
const SH2_1: f32 = -1.0925484305920792;
const SH2_2: f32 = 0.31539156525252005;
const SH2_3: f32 = -1.0925484305920792;
const SH2_4: f32 = 0.5462742152960396;

const SH3_0: f32 = -0.5900435899266435;
const SH3_1: f32 = 2.890611442640554;
const SH3_2: f32 = -0.4570457994644658;
const SH3_3: f32 = 0.3731763325901154;
const SH3_4: f32 = -0.4570457994644658;
const SH3_5: f32 = 1.445305721320277;
const SH3_6: f32 = -0.5900435899266435;

struct Gaussian3d {
    position: vec3<f32>,
    opacity: f32,
    scale: vec3<f32>,
    _pad0: u32,
    rotation: vec4<f32>,
    sh: array<f32, 48>,
};

struct PreprocessOutput {
    conic_opacity: vec4<f32>,   // conic, opacity
    color_radius: vec4<f32>,    // color, radius
    tile_rect: vec4<u32>,
    uv_depth: vec4<f32>, // x, y, depth, pad
};

struct SceneUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    gaussian_count: u32,
    screen_size: vec2<u32>,
    near_far: vec2<f32>,
    tan_fov: vec2<f32>,
    pad: vec2<u32>,
}

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> gaussians: array<Gaussian3d>;

@group(0) @binding(2)
var<storage, read_write> outputs: array<PreprocessOutput>;

@group(0) @binding(3)
var<storage, read_write> tiles_touched: array<u32>;

@group(0) @binding(4)
var<storage, read_write> visible_count: atomic<u32>;

fn ndc_to_pix(ndc: vec2<f32>) -> vec2<f32> {
    let width = f32(scene.screen_size.x);
    let height = f32(scene.screen_size.y);

    return vec2(
        ((ndc.x + 1.0) * width - 1.0) * 0.5,
        ((1.0 - ndc.y) * height - 1.0) * 0.5
    );
}

fn quat_to_mat3_wxyz(q_raw: vec4<f32>) -> mat3x3<f32> {
    let q = normalize(q_raw);

    let w = q.x;
    let x = q.y;
    let y = q.z;
    let z = q.w;

    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;

    let xx = x * x2;
    let yy = y * y2;
    let zz = z * z2;

    let xy = x * y2;
    let xz = x * z2;
    let yz = y * z2;

    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;

    return mat3x3<f32>(
        vec3<f32>(1.0 - (yy + zz), xy + wz, xz - wy),
        vec3<f32>(xy - wz, 1.0 - (xx + zz), yz + wx),
        vec3<f32>(xz + wy, yz - wx, 1.0 - (xx + yy)),
    );
}

fn compute_cov3d(scale_log: vec3<f32>, rotation_wxyz: vec4<f32>) -> mat3x3<f32> {
    // NOTE: The scale is stored as a log scale in PLY
    let sx = exp(scale_log.x);
    let sy = exp(scale_log.y);
    let sz = exp(scale_log.z);

    let R = quat_to_mat3_wxyz(rotation_wxyz);

    // Sigma = R * S^2 * R^T
    let r0 = R[0];
    let r1 = R[1];
    let r2 = R[2];

    let sx2 = sx * sx;
    let sy2 = sy * sy;
    let sz2 = sz * sz;

    let c00 = sx2 * r0.x * r0.x + sy2 * r1.x * r1.x + sz2 * r2.x * r2.x;
    let c01 = sx2 * r0.x * r0.y + sy2 * r1.x * r1.y + sz2 * r2.x * r2.y;
    let c02 = sx2 * r0.x * r0.z + sy2 * r1.x * r1.z + sz2 * r2.x * r2.z;

    let c11 = sx2 * r0.y * r0.y + sy2 * r1.y * r1.y + sz2 * r2.y * r2.y;
    let c12 = sx2 * r0.y * r0.z + sy2 * r1.y * r1.z + sz2 * r2.y * r2.z;

    let c22 = sx2 * r0.z * r0.z + sy2 * r1.z * r1.z + sz2 * r2.z * r2.z;

    return mat3x3<f32>(
        vec3<f32>(c00, c01, c02),
        vec3<f32>(c01, c11, c12),
        vec3<f32>(c02, c12, c22),
    );
}

fn compute_cov2d(cov3d: mat3x3<f32>, view_pos_for_cov: vec3<f32>) -> mat2x2<f32> {
    let J = proj_jacobian(view_pos_for_cov);

    let view_rot = mat3x3<f32>(
        scene.view[0].xyz,
        scene.view[1].xyz,
        scene.view[2].xyz,
    );

    let W = transpose(view_rot);

    let T = W * J;
    var C2 = transpose(T) * cov3d * T;

    C2[0][0] = C2[0][0] + 0.3;
    C2[1][1] = C2[1][1] + 0.3;

    return mat2x2(
        vec2(C2[0][0], C2[0][1]),
        vec2(C2[0][1], C2[1][1]),
    );
}

fn proj_jacobian(view_pos: vec3<f32>) -> mat3x3<f32> {
    let width = f32(scene.screen_size.x);
    let height = f32(scene.screen_size.y);

    let tan_fovx = scene.tan_fov.x;
    let tan_fovy = scene.tan_fov.y;

    let fx = width / (2.0 * tan_fovx);
    let fy = -height / (2.0 * tan_fovy);

    let limx = 1.3 * tan_fovx;
    let limy = 1.3 * tan_fovy;

    let txtz = clamp(view_pos.x / view_pos.z, -limx, limx);
    let tytz = clamp(view_pos.y / view_pos.z, -limy, limy);

    let x = txtz * view_pos.z;
    let y = tytz * view_pos.z;

    let invz = 1.0 / view_pos.z;
    let invz2 = invz * invz;

    return mat3x3<f32>(
        vec3<f32>(fx * invz, 0.0, 0.0),
        vec3<f32>(0.0, fy * invz, 0.0),
        vec3<f32>(-fx * x * invz2, -fy * y * invz2, 0.0),
    );
}

fn sh_coeff(idx: u32, k: u32) -> vec3<f32> {
    let g = gaussians[idx];

    if k == 0u {
        // DC term: f_dc_0/1/2 = R/G/B
        return vec3(g.sh[0], g.sh[1], g.sh[2]);
    }

    // Rest terms stored channel-by-channel: R[0..14], G[0..14], B[0..14]
    let j = k - 1u;
    return vec3(
        g.sh[3u + j],
        g.sh[18u + j],
        g.sh[33u + j]
    );
}

fn eval_sh16(idx: u32, view_pos: vec3<f32>, pos: vec3<f32>) -> vec3<f32> {
    var dir = normalize(pos - view_pos);

    // l=0
    var rgb = SH0 * sh_coeff(idx, 0u);

    // l=1
    rgb += (-SH1 * dir.y) * sh_coeff(idx, 1u);
    rgb += (SH1 * dir.z) * sh_coeff(idx, 2u);
    rgb += (-SH1 * dir.x) * sh_coeff(idx, 3u);

    // l=2
    rgb += (SH2_0 * dir.x * dir.y) * sh_coeff(idx, 4u);
    rgb += (SH2_1 * dir.y * dir.z) * sh_coeff(idx, 5u);
    rgb += (SH2_2 * (2.0 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y)) * sh_coeff(idx, 6u);
    rgb += (SH2_3 * dir.z * dir.x) * sh_coeff(idx, 7u);
    rgb += (SH2_4 * (dir.x * dir.x - dir.y * dir.y)) * sh_coeff(idx, 8u);

    // l=3
    rgb += (SH3_0 * (3.0 * dir.x * dir.x - dir.y * dir.y) * dir.y) * sh_coeff(idx, 9u);
    rgb += (SH3_1 * (dir.x * dir.y * dir.z)) * sh_coeff(idx, 10u);
    rgb += (SH3_2 * (4.0 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y) * dir.y) * sh_coeff(idx, 11u);
    rgb += (SH3_3 * (dir.z * (2.0 * dir.z * dir.z - 3.0 * dir.x * dir.x - 3.0 * dir.y * dir.y))) * sh_coeff(idx, 12u);
    rgb += (SH3_4 * (dir.x * (4.0 * dir.z * dir.z - dir.x * dir.x - dir.y * dir.y))) * sh_coeff(idx, 13u);
    rgb += (SH3_5 * ((dir.x * dir.x - dir.y * dir.y) * dir.z)) * sh_coeff(idx, 14u);
    rgb += (SH3_6 * (dir.x * (dir.x * dir.x - 3.0 * dir.y * dir.y))) * sh_coeff(idx, 15u);

    // Bias +0.5 and clamp
    rgb += vec3(0.5);
    rgb = max(rgb, vec3(0.0));
    return rgb;
}

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

@compute @workgroup_size(256,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= scene.gaussian_count {
        return;
    }
    let g = gaussians[idx];
    let world_pos = vec4<f32>(g.position, 1.0);

    let view_pos4 = scene.view * world_pos;
    let view_pos = vec3<f32>(view_pos4.xy, -view_pos4.z);

    // near clip culling
    if view_pos.z <= scene.near_far.x {
        return;
    }
    let clip = scene.proj * view_pos4;
    if clip.w <= 1e-6 {
        return;
    }

    let ndc = clip.xy / clip.w;
    let uv = ndc_to_pix(ndc);

    let view_pos_for_cov = vec3<f32>(
        view_pos.x,
        view_pos.y,
        view_pos.z,
    );
    let cov3d = compute_cov3d(g.scale, g.rotation);
    let cov2d = compute_cov2d(cov3d, view_pos_for_cov);
    let det_cov2d = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
    if det_cov2d <= 1e-6 {
        return;
    }
    let inv_det = 1.0 / det_cov2d;
    let conic = vec3<f32>(
        cov2d[1][1] * inv_det,
        -cov2d[0][1] * inv_det,
        cov2d[0][0] * inv_det,
    );
    let opacity = 1.0 / (1.0 + exp(-gaussians[idx].opacity));

    let tr = 0.5 * (cov2d[0][0] + cov2d[1][1]);
    let disc = max(0.1, tr * tr - det_cov2d);
    let s = sqrt(disc);
    let lmax = max(tr + s, tr - s);
    let radius = min(ceil(SIGMA_SCALE * sqrt(lmax)), MAX_RADIUS);

    // tile overlap range
    let tiles_x = (u32(scene.screen_size.x) + TILE_W - 1u) / TILE_W;
    let tiles_y = (u32(scene.screen_size.y) + TILE_H - 1u) / TILE_H;
    let min_tx = i32(floor((uv.x - radius) / f32(TILE_W)));
    let min_ty = i32(floor((uv.y - radius) / f32(TILE_H)));
    let max_tx = i32(ceil((uv.x + radius) / f32(TILE_W)));
    let max_ty = i32(ceil((uv.y + radius) / f32(TILE_H)));
    let tile_rect = vec4<u32>(
        u32(clamp(min_tx, 0, i32(tiles_x))),
        u32(clamp(min_ty, 0, i32(tiles_y))),
        u32(clamp(max_tx, 0, i32(tiles_x))),
        u32(clamp(max_ty, 0, i32(tiles_y))),
    );

    // IMPORTANT: 
    //  Count only tiles that may intersect the Gaussian ellipse.
    //  This count is used by prefix scan, so it must match the number of pairs
    //  emitted in duplicate.compute.wgsl exactly.
    var tiles_touched_count = 0u;
    for (var ty = tile_rect.y; ty < tile_rect.w; ty = ty + 1u) {
        for (var tx = tile_rect.x; tx < tile_rect.z; tx = tx + 1u) {
            if tile_may_intersect_conic(uv, conic, tx, ty) {
                tiles_touched_count = tiles_touched_count + 1u;
            }
        }
    }

    if tiles_touched_count == 0u {
        return;
    }

    let rgb = eval_sh16(idx, scene.view_pos.xyz, gaussians[idx].position);

    let visible_idx = atomicAdd(&visible_count, 1u);
    outputs[visible_idx].conic_opacity = vec4<f32>(
        conic.x,
        conic.y,
        conic.z,
        opacity,
    );
    outputs[visible_idx].color_radius = vec4<f32>(
        rgb.x,
        rgb.y,
        rgb.z,
        radius,
    );
    outputs[visible_idx].tile_rect = tile_rect;
    outputs[visible_idx].uv_depth = vec4<f32>(
        uv.x,
        uv.y,
        view_pos.z,
        0.0
    );
    tiles_touched[visible_idx] = tiles_touched_count;
}

