use glam::Vec3;
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::{
    assets,
    camera::{Camera, CameraController, CameraState},
    gpu::context::GpuContext,
    redraw_scheduler::RedrawScheduler,
    renderer::renderer::{RenderStatus, Renderer},
    resources::gaussians::Gaussians,
    scene::{Scene, SceneType},
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

pub enum UserEvent {
    AppReady(AppState),

    #[cfg(target_arch = "wasm32")]
    DroppedFileBytes {
        file_name: String,
        bytes: Vec<u8>,
    },
}

pub struct AppState {
    gpu: GpuContext,

    camera: Camera,
    camera_controller: CameraController,
    camera_state: CameraState,

    fps_timer: Instant,
    frame_count: u32,
    window: Arc<Window>,

    scene: Scene,

    renderer: Renderer,

    last_update_time: Instant,

    is_focused: bool,
    is_occluded: bool,

    redraw_scheduler: RedrawScheduler,
}

impl AppState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let gpu = GpuContext::new(Arc::clone(&window)).await?;

        let surface_size = gpu.surface_size();
        let camera = Camera {
            eye: (10.0, 5.0, 10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vec3::Y,
            aspect: surface_size.width as f32 / surface_size.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };
        let mut camera_controller = CameraController::new(5.0);
        camera_controller.sync_from_camera(&camera);

        let scene_type = SceneType::Gaussian3d;
        let scene = Scene::new(scene_type);

        let gaussians = Gaussians::Gaussian3d(Vec::new());
        let renderer = Renderer::new(&gpu, &camera, &gaussians, scene_type);

        let redraw_scheduler = RedrawScheduler::new(Arc::clone(&window));

        Ok(Self {
            gpu,
            camera,
            camera_controller,
            camera_state: CameraState::Idle,
            fps_timer: Instant::now(),
            frame_count: 0,
            window,
            scene,
            renderer,
            last_update_time: Instant::now(),
            is_focused: true,
            is_occluded: false,
            redraw_scheduler,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32, scale_factor: f64) {
        if !self.gpu.resize_surface(width, height, scale_factor) {
            return;
        }

        let surface_size = self.gpu.surface_size();
        let width = surface_size.width;
        let height = surface_size.height;
        log::info!("resize: ({width}, {height})");

        self.camera.update_aspect(width, height);

        self.renderer
            .resize(&mut self.gpu, &self.scene, &self.camera);

        self.redraw_scheduler.request_redraw();
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt_sec = (now - self.last_update_time).as_secs_f32();
        self.last_update_time = now;

        self.camera_state = self
            .camera_controller
            .update_camera(&mut self.camera, dt_sec);

        self.scene.update(dt_sec);

        self.renderer.update(&self.gpu, &self.scene, &self.camera);
    }

    pub fn render(&mut self) -> anyhow::Result<RenderStatus> {
        let status = self.renderer.render(&self.gpu, &self.redraw_scheduler)?;

        if status == RenderStatus::Submitted {
            self.update_fps_counter();
        }
        Ok(status)
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => {
                let changed = self.camera_controller.handle_key(code, is_pressed);

                if changed {
                    if is_pressed {
                        self.last_update_time = Instant::now();
                    }

                    self.redraw_scheduler.request_redraw();
                }
            }
        }
    }

    fn update_fps_counter(&mut self) {
        self.frame_count += 1;

        let elapsed = self.fps_timer.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let fps = self.frame_count as f64 / elapsed.as_secs_f64();
            log::info!("FPS: {:.1}", fps);
            log::info!(
                "frame: {:.1} ms",
                elapsed.as_millis() as f32 / self.frame_count as f32
            );
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }

    pub fn should_request_redraw(&self) -> bool {
        self.camera_state == CameraState::Active || self.scene.scene_type.is_dynamic()
    }

    pub fn replace_gaussians(&mut self, gaussians: Gaussians) -> anyhow::Result<()> {
        let new_scene_type = match &gaussians {
            Gaussians::Gaussian3d(_) => SceneType::Gaussian3d,
            Gaussians::Gaussian4d(_) => SceneType::Gaussian4d,
        };

        self.renderer
            .replace_gaussians(&mut self.gpu, &gaussians, new_scene_type);

        let gaussian_count = gaussians.len() as u32;
        self.scene.replace_gaussians(new_scene_type, gaussian_count);

        Ok(())
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<UserEvent>>,
    state: Option<AppState>,
    scale_factor: f64,
    window_size: winit::dpi::PhysicalSize<u32>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<UserEvent>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
            scale_factor: 1.0,
            window_size: winit::dpi::PhysicalSize::new(1, 1),
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes = Window::default_attributes();

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let width = window.inner_size().width;
        let height = window.inner_size().height;
        self.window_size = winit::dpi::PhysicalSize::new(width, height);
        log::info!("resume window size: ({width}, {height})");

        self.scale_factor = window.scale_factor();

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.state = Some(pollster::block_on(AppState::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            if let Some(proxy) = self.proxy.take() {
                install_web_drag_and_drop(proxy.clone());
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(UserEvent::AppReady(
                                AppState::new(window)
                                    .await
                                    .expect("Unable to create canvas!!")
                            ))
                            .is_ok()
                    )
                });
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height, self.scale_factor);
                self.window_size = winit::dpi::PhysicalSize::new(size.width, size.height);
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                state.resize(
                    self.window_size.width,
                    self.window_size.height,
                    scale_factor,
                );
                self.scale_factor = scale_factor;
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::DroppedFile(path) => {
                let format = assets::format::FileFormat::from_path(&path);
                let Some(format) = format else {
                    log::error!("unsupported file format: {path:?}");
                    return;
                };
                match assets::loader::load_gaussians_from_path(format, &path).and_then(
                    |gaussians| {
                        let gaussian_count = gaussians.len();
                        state.replace_gaussians(gaussians)?;
                        Ok(gaussian_count)
                    },
                ) {
                    Ok(gaussian_count) => {
                        log::info!("loaded dropped file: {path:?}");
                        log::info!("gaussian count: {gaussian_count}");
                        state.redraw_scheduler.request_redraw();
                    }
                    Err(e) => {
                        log::error!("{e:?}: {path:?}");
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if state.is_occluded || !state.is_focused {
                    return;
                }

                if !state.redraw_scheduler.try_begin_frame() {
                    return;
                }

                state.update();

                match state.render() {
                    Ok(RenderStatus::Submitted) => {}

                    Ok(RenderStatus::Skipped) => {
                        state.redraw_scheduler.cancel_frame();
                    }

                    Err(error) => {
                        state.redraw_scheduler.cancel_frame();

                        log::error!("{error}");
                        event_loop.exit();
                        return;
                    }
                }

                if state.should_request_redraw() {
                    state.redraw_scheduler.request_redraw();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::Focused(focused) => {
                state.is_focused = focused;

                if focused {
                    state.last_update_time = Instant::now();
                    state.redraw_scheduler.request_redraw();
                } else {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if let Err(error) = state.gpu.poll_once() {
                            log::error!("failed to poll GPU device: {error:?}");
                        }
                    }
                }
            }

            WindowEvent::Occluded(occluded) => {
                state.is_occluded = occluded;

                if !occluded {
                    state.last_update_time = Instant::now();
                    state.redraw_scheduler.request_redraw();
                } else {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        if let Err(error) = state.gpu.poll_once() {
                            log::error!("failed to poll GPU device: {error:?}");
                        }
                    }
                }
            }
            _ => {}
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::AppReady(mut state) => {
                #[cfg(target_arch = "wasm32")]
                {
                    state.redraw_scheduler.request_redraw();
                    state.resize(
                        state.window.inner_size().width,
                        state.window.inner_size().height,
                        state.window.scale_factor(),
                    );

                    let scale_factor = state.window.scale_factor();
                    log::info!("{scale_factor}");
                }

                self.state = Some(state);
            }

            #[cfg(target_arch = "wasm32")]
            UserEvent::DroppedFileBytes { file_name, bytes } => {
                let Some(state) = &mut self.state else {
                    return;
                };

                let format = assets::format::FileFormat::from_file_name(&file_name);
                let Some(format) = format else {
                    log::error!("unsupported file format: {file_name}");
                    return;
                };

                match assets::loader::load_gaussians_from_bytes(format, &bytes).and_then(
                    |gaussians| {
                        let gaussian_count = gaussians.len();
                        state.replace_gaussians(gaussians)?;
                        Ok(gaussian_count)
                    },
                ) {
                    Ok(gaussian_count) => {
                        log::info!("loaded dropped file: {file_name}");
                        log::info!("gaussian count: {gaussian_count}");
                        state.redraw_scheduler.request_redraw();
                    }
                    Err(e) => {
                        log::error!("{e:?}: {file_name}");
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn install_web_drag_and_drop(proxy: winit::event_loop::EventLoopProxy<UserEvent>) {
    use wasm_bindgen::JsCast;
    use wasm_bindgen::closure::Closure;
    use web_sys::{DragEvent, FileReader};

    let window = wgpu::web_sys::window().unwrap_throw();
    let document = window.document().unwrap_throw();
    let body = document.body().unwrap_throw();

    let dragover_handler = Closure::<dyn FnMut(DragEvent)>::new(|event: DragEvent| {
        event.prevent_default();
    });
    body.set_ondragover(Some(dragover_handler.as_ref().unchecked_ref()));
    dragover_handler.forget();

    let proxy = std::rc::Rc::new(proxy);
    let drop_handler = Closure::<dyn FnMut(DragEvent)>::new(move |event: DragEvent| {
        event.prevent_default();
        let proxy = proxy.clone();
        if let Some(dt) = event.data_transfer() {
            if let Some(files) = dt.files() {
                if let Some(file) = files.item(0) {
                    let reader = FileReader::new().unwrap_throw();
                    let reader_clone = reader.clone();
                    let file_name = file.name();
                    let onload = Closure::<dyn FnMut()>::new(move || {
                        if let Ok(result) = reader_clone.result() {
                            if let Some(ab) = result.dyn_ref::<js_sys::ArrayBuffer>() {
                                let bytes = js_sys::Uint8Array::new(ab).to_vec();
                                let _ = proxy.send_event(UserEvent::DroppedFileBytes {
                                    file_name: file_name.clone(),
                                    bytes,
                                });
                            }
                        }
                    });
                    reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                    onload.forget();
                    reader.read_as_array_buffer(&file).unwrap_throw();
                }
            }
        }
    });
    body.set_ondrop(Some(drop_handler.as_ref().unchecked_ref()));
    drop_handler.forget();
}
