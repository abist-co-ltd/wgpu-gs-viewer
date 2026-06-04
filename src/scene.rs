use crate::camera;
use glam::*;

pub const SCREEN_WIDTH: u32 = 1280;
pub const SCREEN_HEIGHT: u32 = 720;

pub const TIME_SPEED: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneType {
    Gaussian3d,
    Gaussian4d,
}

impl SceneType {
    pub fn is_dynamic(self) -> bool {
        self == Self::Gaussian4d
    }
}

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
    pub fn new() -> Self {
        Self {
            view: Mat4::IDENTITY.to_cols_array_2d(),
            proj: Mat4::IDENTITY.to_cols_array_2d(),
            view_pos: [0.0, 0.0, 0.0],
            gaussian_count: 0,
            screen_size: [SCREEN_WIDTH, SCREEN_HEIGHT],
            near_far: [0.01, 100.0],
            tan_fov: [0.0, 0.0],
            time: 0.0,
            _pad0: 0,
        }
    }

    pub fn update_camera(&mut self, camera: &camera::Camera) {
        self.view = camera.build_view_matrix().to_cols_array_2d();
        self.proj = camera.build_projection_matrix().to_cols_array_2d();
        self.view_pos = camera.eye.into();
        self.near_far = [camera.znear, camera.zfar];
        let tan_fovy = (camera.fovy.to_radians() * 0.5).tan();
        let tan_fovx = tan_fovy * camera.aspect;
        self.tan_fov = [tan_fovx, tan_fovy];
    }

    pub fn update_gaussian_count(&mut self, gaussian_count: u32) {
        self.gaussian_count = gaussian_count;
    }

    pub fn update_time(&mut self, dt: f32) {
        self.time = (self.time + dt * TIME_SPEED).rem_euclid(1.0);
    }
}
