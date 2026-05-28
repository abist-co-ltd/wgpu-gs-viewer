use glam::*;
use winit::keyboard::KeyCode;

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }
    pub fn build_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum CameraState {
    Idle,
    Active,
}

// orbit camera
pub struct CameraController {
    rotate_speed: f32,
    zoom_speed: f32,

    rotation: Quat,
    radius: f32,

    is_yaw_left_pressed: bool,
    is_yaw_right_pressed: bool,
    is_pitch_up_pressed: bool,
    is_pitch_down_pressed: bool,

    is_roll_left_pressed: bool,
    is_roll_right_pressed: bool,

    is_zoom_in_pressed: bool,
    is_zoom_out_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            rotate_speed: speed * 0.25,
            zoom_speed: speed,

            rotation: Quat::IDENTITY,
            radius: 10.0,

            is_yaw_left_pressed: false,
            is_yaw_right_pressed: false,
            is_pitch_up_pressed: false,
            is_pitch_down_pressed: false,

            is_roll_left_pressed: false,
            is_roll_right_pressed: false,

            is_zoom_in_pressed: false,
            is_zoom_out_pressed: false,
        }
    }

    pub fn sync_from_camera(&mut self, camera: &Camera) {
        let offset = camera.eye - camera.target;
        let radius = offset.length();

        if radius < 1e-5 {
            self.radius = 1.0;
            self.rotation = Quat::IDENTITY;
            return;
        }

        self.radius = radius;

        let dir = offset / radius;
        self.rotation = Quat::from_rotation_arc(Vec3::Z, dir).normalize();
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                let changed = self.is_yaw_left_pressed != is_pressed;
                self.is_yaw_left_pressed = is_pressed;
                changed
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                let changed = self.is_yaw_right_pressed != is_pressed;
                self.is_yaw_right_pressed = is_pressed;
                changed
            }
            KeyCode::KeyW | KeyCode::ArrowUp => {
                let changed = self.is_pitch_up_pressed != is_pressed;
                self.is_pitch_up_pressed = is_pressed;
                changed
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                let changed = self.is_pitch_down_pressed != is_pressed;
                self.is_pitch_down_pressed = is_pressed;
                changed
            }
            KeyCode::KeyQ => {
                let changed = self.is_zoom_in_pressed != is_pressed;
                self.is_zoom_in_pressed = is_pressed;
                changed
            }
            KeyCode::KeyE => {
                let changed = self.is_zoom_out_pressed != is_pressed;
                self.is_zoom_out_pressed = is_pressed;
                changed
            }
            KeyCode::KeyZ => {
                let changed = self.is_roll_left_pressed != is_pressed;
                self.is_roll_left_pressed = is_pressed;
                changed
            }
            KeyCode::KeyC => {
                let changed = self.is_roll_right_pressed != is_pressed;
                self.is_roll_right_pressed = is_pressed;
                changed
            }
            _ => false,
        }
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt_sec: f32) -> CameraState {
        let yaw_input = (self.is_yaw_right_pressed as i32 - self.is_yaw_left_pressed as i32) as f32;

        let pitch_input =
            (self.is_pitch_up_pressed as i32 - self.is_pitch_down_pressed as i32) as f32;

        let roll_input =
            (self.is_roll_right_pressed as i32 - self.is_roll_left_pressed as i32) as f32;

        let zoom_input = (self.is_zoom_out_pressed as i32 - self.is_zoom_in_pressed as i32) as f32;

        let mut changed = false;

        if yaw_input != 0.0 {
            let yaw = yaw_input * self.rotate_speed * dt_sec;

            let up = self.rotation * Vec3::Y;
            let q = Quat::from_axis_angle(up.normalize(), yaw);

            self.rotation = (q * self.rotation).normalize();
            changed = true;
        }

        if pitch_input != 0.0 {
            let pitch = pitch_input * self.rotate_speed * dt_sec;

            let right = self.rotation * Vec3::X;
            let q = Quat::from_axis_angle(right.normalize(), pitch);

            self.rotation = (q * self.rotation).normalize();
            changed = true;
        }

        if roll_input != 0.0 {
            let roll = roll_input * self.rotate_speed * dt_sec;

            let forward = self.rotation * -Vec3::Z;
            let q = Quat::from_axis_angle(forward.normalize(), roll);

            self.rotation = (q * self.rotation).normalize();
            changed = true;
        }

        if zoom_input != 0.0 {
            self.radius += zoom_input * self.zoom_speed * dt_sec;
            self.radius = self.radius.clamp(0.1, 10000.0);
            changed = true;
        }

        if !changed {
            return CameraState::Idle;
        }

        camera.target = Vec3::ZERO;

        let offset = self.rotation * Vec3::new(0.0, 0.0, self.radius);

        camera.eye = camera.target + offset;
        camera.up = (self.rotation * Vec3::Y).normalize();

        CameraState::Active
    }
}
