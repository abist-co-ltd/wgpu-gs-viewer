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

@group(0) @binding(0)
var<uniform> scene: SceneUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = vec4<f32>(input.position, 1.0);
    out.clip_position = scene.proj * scene.view * world_pos;
    out.color = input.color;

    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}