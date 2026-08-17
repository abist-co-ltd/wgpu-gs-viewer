const MAX_AXIS_LENGTH: f32 = 1024.0;

const ALPHA_CLIP: f32 = 1.0 / 255.0;

const PI: f32 = 3.14159265;
const MIN_CONTRIBUTION: f32 = 3.0;

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
    center_depth_extent: vec4<f32>,
    elipse_axis: vec4<f32>, // major_axis, minor_axis
    color_opacity: vec4<f32>,
};

struct SceneUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_pos: vec3<f32>,
    gaussian_count: u32,
    screen_size: vec2<u32>,
    near_far: vec2<f32>,
    tan_fov: vec2<f32>,
    time: f32,
    pad: u32,
};

struct DepthRange {
    min_bits: atomic<u32>,
    max_bits: atomic<u32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> gaussians: array<Gaussian3d>;

@group(0) @binding(2)
var<storage, read_write> outputs: array<PreprocessOutput>;

@group(0) @binding(3)
var<storage, read_write> sort_keys: array<u32>;

@group(0) @binding(4)
var<storage, read_write> sort_values: array<u32>;

@group(0) @binding(5)
var<storage, read_write> visible_count: atomic<u32>;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

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

    let flip_z = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, -1.0),
    );

    let W = flip_z * view_rot;

    let cov_camera = W * cov3d * transpose(W);
    var C2 = J * cov_camera * transpose(J);
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

fn ellipse_outside_screen(
    center: vec2<f32>,
    major_axis: vec2<f32>,
    minor_axis: vec2<f32>,
    extent: f32
) -> bool {
    let radius = extent * vec2<f32>(
        sqrt(
            major_axis.x * major_axis.x +
            minor_axis.x * minor_axis.x
        ),
        sqrt(
            major_axis.y * major_axis.y +
            minor_axis.y * minor_axis.y
        ),
    );

    let min_pos = center - radius;
    let max_pos = center + radius;

    let screen = vec2<f32>(scene.screen_size);

    return max_pos.x < 0.0 ||
        max_pos.y < 0.0 ||
        min_pos.x >= screen.x ||
        min_pos.y >= screen.y;
}

fn principal_axis(
    cov2d: mat2x2<f32>,
    lambda: f32,
) -> vec2<f32> {
    let candidate = vec2<f32>(cov2d[0][1], lambda - cov2d[0][0]);

    if dot(candidate, candidate) > 1e-12 {
        return normalize(candidate);
    }

    if cov2d[0][0] >= cov2d[1][1] {
        return vec2<f32>(1.0, 0.0);
    }

    return vec2<f32>(0.0, 1.0);
}

@compute @workgroup_size(256, 1, 1)
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

    let tr = 0.5 * (cov2d[0][0] + cov2d[1][1]);
    let disc = max(0.0, tr * tr - det_cov2d);
    let s = sqrt(disc);
    let lambda_major = tr + s;
    let lambda_minor = tr - s;

    if lambda_minor <= 0.0 {
        return;
    }

    let major_direction = principal_axis(
        cov2d,
        lambda_major,
    );

    let minor_direction = vec2<f32>(
        major_direction.y,
        -major_direction.x,
    );

    // Fragment Shader:
    //     alpha = exp(-dot(local, local)) * opacity
    //
    // Therefore, we use sqrt(2 * lambda) as the axis length.
    let major_length = min(
        sqrt(lambda_major),
        MAX_AXIS_LENGTH,
    );

    let minor_length = min(
        sqrt(lambda_minor),
        MAX_AXIS_LENGTH,
    );

    let major_axis = major_direction * major_length;
    let minor_axis = minor_direction * minor_length;

    let opacity = sigmoid(g.opacity);
    let rgb = eval_sh16(idx, scene.view_pos.xyz, g.position);

    let extent = select(
        0.0,
        sqrt(2.0 * log(opacity / ALPHA_CLIP)),
        opacity > ALPHA_CLIP,
    );

    let major_radius_px = length(major_axis.xy) * extent;
    let minor_radius_px = length(minor_axis.xy) * extent;

    if ellipse_outside_screen(
        uv,
        major_axis,
        minor_axis,
        extent
    ) {
        return;
    }

    let projected_area = PI * major_radius_px * minor_radius_px;

    if opacity * projected_area < MIN_CONTRIBUTION {
        return;
    }

    let visible_idx = atomicAdd(&visible_count, 1u);

    outputs[visible_idx].center_depth_extent = vec4<f32>(
        uv.x,
        uv.y,
        view_pos.z,
        extent,
    );

    outputs[visible_idx].elipse_axis = vec4<f32>(
        major_axis.x,
        major_axis.y,
        minor_axis.x,
        minor_axis.y,
    );

    outputs[visible_idx].color_opacity = vec4<f32>(
        rgb,
        opacity,
    );

    // radix sort input
    sort_keys[visible_idx] = bitcast<u32>(view_pos.z);
    sort_values[visible_idx] = visible_idx;
}