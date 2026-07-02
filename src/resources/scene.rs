use crate::camera::Camera;
use crate::gpu::context::GpuContext;
use crate::scene::Scene;
use glam::*;

// TODO: Scene/Cameraに分離
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SceneUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_pos: [f32; 3],
    gaussian_count: u32,
    screen_size: [u32; 2],
    near_far: [f32; 2],
    tan_fov: [f32; 2],
    time: f32,
    _pad0: u32,
}

impl SceneUniform {
    fn new() -> Self {
        Self {
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            view_pos: [0.0, 0.0, 0.0],
            gaussian_count: 0,
            screen_size: [1, 1],
            near_far: [0.01, 100.0],
            tan_fov: [0.0, 0.0],
            time: 0.0,
            _pad0: 0,
        }
    }

    fn update_screen_size(&mut self, screen_size: [u32; 2]) {
        self.screen_size = screen_size;
    }

    fn update_camera(&mut self, camera: &Camera) {
        self.view = camera.build_view_matrix().to_cols_array_2d();
        self.proj = camera.build_projection_matrix().to_cols_array_2d();
        self.view_pos = camera.eye.into();
        self.near_far = [camera.znear, camera.zfar];
        let tan_fovy = (camera.fovy.to_radians() * 0.5).tan();
        let tan_fovx = tan_fovy * camera.aspect;
        self.tan_fov = [tan_fovx, tan_fovy];
    }

    fn update(&mut self, camera: &Camera, scene: &Scene, screen_size: [u32; 2]) {
        self.update_screen_size(screen_size);
        self.update_camera(camera);
        self.gaussian_count = scene.gaussian_count;
        self.time = scene.time;
    }
}

pub struct SceneResource {
    uniform: SceneUniform,
    pub uniform_buffer: wgpu::Buffer,
}

impl SceneResource {
    pub fn new(gpu: &GpuContext, camera: &Camera) -> Self {
        let mut uniform = SceneUniform::new();

        let surface_size = gpu.surface_size();
        let width = surface_size.width.max(1);
        let height = surface_size.height.max(1);

        uniform.update_screen_size([width, height]);
        uniform.update_camera(camera);

        let uniform_buffer = gpu.create_buffer_init(
            "Scene Uniform Buffer",
            bytemuck::bytes_of(&uniform),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        Self {
            uniform,
            uniform_buffer,
        }
    }

    pub fn update(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        let surface_size = gpu.surface_size();
        let width = surface_size.width.max(1);
        let height = surface_size.height.max(1);

        self.uniform.update(camera, scene, [width, height]);

        self.upload(gpu);
    }

    fn upload(&self, gpu: &GpuContext) {
        gpu.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
    }
}
