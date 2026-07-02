use glam::*;

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

pub struct Scene {
    pub scene_type: SceneType,
    pub gaussian_count: u32,
    pub time: f32,
}

impl Scene {
    pub fn new(scene_type: SceneType) -> Self {
        Self {
            scene_type,
            gaussian_count: 0,
            time: 0.0,
        }
    }

    pub fn replace_gaussians(&mut self, scene_type: SceneType, gaussian_count: u32) {
        self.scene_type = scene_type;
        self.gaussian_count = gaussian_count;
        self.time = 0.0;
    }

    pub fn update(&mut self, dt: f32) {
        if self.scene_type.is_dynamic() {
            self.time = (self.time + dt * TIME_SPEED).rem_euclid(1.0);
        }
    }
}
