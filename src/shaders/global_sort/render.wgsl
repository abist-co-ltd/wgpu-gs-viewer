const ALPHA_CLIP: f32 = 1.0 / 255.0;

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

struct PreprocessOutput {
    center_depth_extent: vec4<f32>,
    elipse_axis: vec4<f32>,
    color_opacity: vec4<f32>,
};

struct VertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    local_position: vec2<f32>,
    @location(1)
    color_opacity: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

@group(0) @binding(1)
var<storage, read> preprocess_outputs: array<PreprocessOutput>;

@group(0) @binding(2)
var<storage, read> sorted_values: array<u32>;

fn quad_position(vertex_index: u32, extent: f32) -> vec2<f32> {
    let positions = array<vec2<f32>, 4>(
        vec2<f32>(-extent, -extent),
        vec2<f32>(extent, -extent),
        vec2<f32>(-extent, extent),
        vec2<f32>(extent, extent),
    );

    return positions[vertex_index];
}

fn pixel_to_ndc(pixel: vec2<f32>) -> vec2<f32> {
    let screen = vec2<f32>(scene.screen_size);

    return vec2<f32>(
        pixel.x / screen.x * 2.0 - 1.0,
        1.0 - pixel.y / screen.y * 2.0,
    );
}

@vertex
fn vs_main(
    @builtin(vertex_index)
    vertex_index: u32,
    @builtin(instance_index)
    instance_index: u32,
) -> VertexOutput {
    let preprocess_index = sorted_values[instance_index];

    let gaussian = preprocess_outputs[preprocess_index];

    let extent = gaussian.center_depth_extent.w;
    let local = quad_position(vertex_index, extent);

    let pixel_position = gaussian.center_depth_extent.xy + local.x * gaussian.elipse_axis.xy + local.y * gaussian.elipse_axis.zw;

    var output: VertexOutput;
    output.position = vec4<f32>(
        pixel_to_ndc(pixel_position),
        0.0,
        1.0,
    );
    output.local_position = local;
    output.color_opacity = gaussian.color_opacity;

    return output;
}

@fragment
fn fs_main(
    input: VertexOutput,
) -> @location(0) vec4<f32> {
    let exponent = -0.5 * dot(input.local_position, input.local_position);

    let alpha = exp(exponent) * input.color_opacity.a;

    if alpha < ALPHA_CLIP {
        discard;
    }

    return vec4<f32>(
        input.color_opacity.rgb * alpha,
        alpha,
    );
}