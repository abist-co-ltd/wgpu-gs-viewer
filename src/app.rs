use glam::*;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::{assets, camera, gaussian, passes, scene};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
const MAX_FRAMES_IN_FLIGHT: u32 = 1;

pub enum UserEvent {
    AppReady(AppState),

    #[cfg(target_arch = "wasm32")]
    DroppedFileBytes {
        file_name: String,
        bytes: Vec<u8>,
    },
}

pub struct AppState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    render_texture_sampler: wgpu::Sampler,
    camera: camera::Camera,
    camera_controller: camera::CameraController,
    camera_state: camera::CameraState,
    scene_uniform: scene::SceneUniform,
    scene_uniform_buffer: wgpu::Buffer,
    start_time: Instant,
    fps_timer: Instant,
    frame_count: u32,
    window: Arc<Window>,
    clear_color: wgpu::Color,

    scene_type: scene::SceneType,

    resources: gaussian::GaussianResources,

    preprocess_pass: passes::preprocess::PreprocessPass,
    prefix_scan_pass: passes::prefix_scan::PrefixScanPass,
    duplicate_pass: passes::duplicate::DuplicatePass,
    radix_sort_pass: passes::radix_sort::RadixSortPass,
    tile_range_pass: passes::tile_range::TileRangePass,
    tile_render_pass: passes::tile_render::TileRenderPass,
    screen_blit_pass: passes::screen_blit::ScreenBlitPass,
    axis_pass: passes::axis::AxisPass,

    scene_dirty: bool,
    last_update_time: Instant,

    is_focused: bool,
    is_occluded: bool,

    #[cfg(target_arch = "wasm32")]
    frames_in_flight: Arc<AtomicU32>,
    #[cfg(target_arch = "wasm32")]
    wants_redraw_after_gpu_done: Arc<AtomicBool>,
}

impl AppState {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            flags: Default::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        #[cfg(not(target_arch = "wasm32"))]
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .filter(|adapter| adapter.is_surface_supported(&surface))
            .next()
            .unwrap();
        #[cfg(target_arch = "wasm32")]
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await?;

        let adapter_limits = adapter.limits();
        let required_limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits {
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                ..wgpu::Limits::downlevel_webgl2_defaults()
            }
        } else {
            wgpu::Limits {
                max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                max_buffer_size: adapter_limits.max_buffer_size,
                ..wgpu::Limits::default()
            }
        };

        log::info!(
            "adapter max_storage_buffer_binding_size={} MB, max_buffer_size={} MB",
            adapter_limits.max_storage_buffer_binding_size / 1024 / 1024,
            adapter_limits.max_buffer_size / 1024 / 1024,
        );

        log::info!(
            "required max_storage_buffer_binding_size={} MB, max_buffer_size={} MB",
            required_limits.max_storage_buffer_binding_size / 1024 / 1024,
            required_limits.max_buffer_size / 1024 / 1024,
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Rgba8Unorm)
            .or_else(|| {
                surface_caps
                    .formats
                    .iter()
                    .copied()
                    .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm)
            })
            .or_else(|| surface_caps.formats.iter().copied().find(|f| !f.is_srgb()))
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render texture"),
            size: wgpu::Extent3d {
                width: scene::SCREEN_WIDTH,
                height: scene::SCREEN_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let render_texture_view =
            render_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let render_texture_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let camera = camera::Camera {
            eye: (10.0, 5.0, 10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: Vec3::Y,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 10000.0,
        };
        let mut camera_controller = camera::CameraController::new(5.0);
        camera_controller.sync_from_camera(&camera);

        let mut scene_uniform = scene::SceneUniform::new();
        scene_uniform.update_camera(&camera);
        let scene_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Scene Uniform Buffer"),
            contents: bytemuck::cast_slice(&[scene_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let scene_type = scene::SceneType::Gaussian3d;

        let resources =
            gaussian::GaussianResources::new(&device, &gaussian::Gaussians::Gaussian3d(Vec::new()));
        scene_uniform.update_gaussian_count(resources.gaussian_count);

        let preprocess_pass = passes::preprocess::PreprocessPass::new(
            &device,
            &scene_uniform_buffer,
            &resources,
            scene_type,
        );
        let prefix_scan_pass = passes::prefix_scan::PrefixScanPass::new(&device, &resources);
        let duplicate_pass = passes::duplicate::DuplicatePass::new(&device, &resources);
        let radix_sort_pass = passes::radix_sort::RadixSortPass::new(&device, &resources);
        let tile_range_pass = passes::tile_range::TileRangePass::new(&device, &resources);
        let tile_render_pass = passes::tile_render::TileRenderPass::new(
            &device,
            &scene_uniform_buffer,
            &render_texture_view,
            &resources,
        );
        let screen_blit_pass = passes::screen_blit::ScreenBlitPass::new(
            &device,
            &render_texture_view,
            &render_texture_sampler,
            config.format,
            wgpu::TextureFormat::Depth32Float,
        );
        let axis_pass = passes::axis::AxisPass::new(&device, &scene_uniform_buffer, config.format);

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: true,
            depth_texture,
            depth_texture_view,
            render_texture,
            render_texture_view,
            render_texture_sampler,
            camera,
            camera_controller,
            camera_state: camera::CameraState::Idle,
            scene_uniform,
            scene_uniform_buffer,
            start_time: Instant::now(),
            fps_timer: Instant::now(),
            frame_count: 0,
            window,
            clear_color: wgpu::Color {
                r: 0.1,
                g: 0.2,
                b: 0.3,
                a: 1.0,
            },
            scene_type,
            resources,
            preprocess_pass,
            prefix_scan_pass,
            duplicate_pass,
            radix_sort_pass,
            tile_range_pass,
            tile_render_pass,
            screen_blit_pass,
            axis_pass,
            scene_dirty: true,
            last_update_time: Instant::now(),
            is_focused: true,
            is_occluded: false,
            #[cfg(target_arch = "wasm32")]
            frames_in_flight: Arc::new(AtomicU32::new(0)),
            #[cfg(target_arch = "wasm32")]
            wants_redraw_after_gpu_done: Arc::new(AtomicBool::new(false)),
        })
    }

    #[cfg(target_arch = "wasm32")]
    pub fn can_submit_frame(&self) -> bool {
        self.frames_in_flight.load(Ordering::Relaxed) < MAX_FRAMES_IN_FLIGHT
    }

    #[cfg(target_arch = "wasm32")]
    pub fn request_redraw_or_defer(&self) {
        if self.can_submit_frame() {
            self.window.request_redraw();
        } else {
            self.wants_redraw_after_gpu_done
                .store(true, Ordering::Relaxed);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.config.width == width && self.config.height == height {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.is_surface_configured = true;

        self.camera.aspect = self.config.width as f32 / self.config.height as f32;

        self.depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: self.config.width.max(1),
                height: self.config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.depth_texture_view = self
            .depth_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.scene_uniform.update_camera(&self.camera);
        self.scene_uniform
            .update_gaussian_count(self.resources.gaussian_count);

        self.queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.scene_uniform]),
        );

        self.scene_dirty = true;
        #[cfg(target_arch = "wasm32")]
        self.request_redraw_or_defer();

        #[cfg(not(target_arch = "wasm32"))]
        self.window.request_redraw();
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt_sec = (now - self.last_update_time).as_secs_f32();
        self.last_update_time = now;

        self.camera_state = self
            .camera_controller
            .update_camera(&mut self.camera, dt_sec);

        if self.camera_state == camera::CameraState::Active {
            self.scene_dirty = true;
        }

        self.scene_uniform.update_camera(&self.camera);
        self.scene_uniform
            .update_gaussian_count(self.resources.gaussian_count);
        self.scene_uniform.update_time(dt_sec);

        self.queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.scene_uniform]),
        );
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        if !self.is_surface_configured {
            return Ok(());
        }

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.surface.configure(&self.device, &self.config);
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                anyhow::bail!("Lost device");
            }
        };

        // clear
        self.queue.write_buffer(
            &self.resources.visible_count_buffer,
            0,
            bytemuck::bytes_of(&0u32),
        );

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        if self.scene_dirty || self.scene_type.is_dynamic() {
            self.preprocess_pass
                .encode(&mut encoder, self.resources.gaussian_count);
            self.prefix_scan_pass.encode(&mut encoder, &self.resources);
            self.duplicate_pass.encode(&mut encoder, &self.resources);
            self.radix_sort_pass.encode(&mut encoder, &self.resources);
            self.tile_range_pass.encode(&mut encoder, &self.resources);
            self.tile_render_pass.encode(
                &mut encoder,
                self.resources.tiles_width,
                self.resources.tiles_height,
            );
            self.scene_dirty = false;
        }

        self.screen_blit_pass
            .encode(&mut encoder, &view, &self.depth_texture_view);
        self.axis_pass.encode(&mut encoder, &view);

        #[cfg(target_arch = "wasm32")]
        {
            let command_buffer = encoder.finish();
            self.frames_in_flight.fetch_add(1, Ordering::Relaxed);

            let frames_in_flight = self.frames_in_flight.clone();
            let wants_redraw_after_gpu_done = self.wants_redraw_after_gpu_done.clone();
            let window = self.window.clone();

            self.queue.submit(std::iter::once(command_buffer));

            self.queue.on_submitted_work_done(move || {
                frames_in_flight
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                        Some(current.saturating_sub(1))
                    })
                    .ok();

                if wants_redraw_after_gpu_done.swap(false, Ordering::Relaxed) {
                    window.request_redraw();
                }
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        self.queue.submit(std::iter::once(encoder.finish()));

        output.present();
        self.update_fps_counter();

        Ok(())
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

                    #[cfg(target_arch = "wasm32")]
                    self.request_redraw_or_defer();

                    #[cfg(not(target_arch = "wasm32"))]
                    self.window.request_redraw();
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
            self.frame_count = 0;
            self.fps_timer = Instant::now();
        }
    }

    pub fn should_request_redraw(&self) -> bool {
        self.scene_dirty
            || self.camera_state == camera::CameraState::Active
            || self.scene_type.is_dynamic()
    }

    pub fn replace_gaussians(&mut self, gaussians: gaussian::Gaussians) -> anyhow::Result<()> {
        let new_scene_type = match &gaussians {
            gaussian::Gaussians::Gaussian3d(_) => scene::SceneType::Gaussian3d,
            gaussian::Gaussians::Gaussian4d(_) => scene::SceneType::Gaussian4d,
        };
        self.resources = gaussian::GaussianResources::new(&self.device, &gaussians);
        if self.scene_type == new_scene_type {
            self.preprocess_pass.recreate_bind_group(
                &self.device,
                &self.scene_uniform_buffer,
                &self.resources,
            );
        } else {
            self.preprocess_pass = passes::preprocess::PreprocessPass::new(
                &self.device,
                &self.scene_uniform_buffer,
                &self.resources,
                new_scene_type,
            );
        }
        self.preprocess_pass.recreate_bind_group(
            &self.device,
            &self.scene_uniform_buffer,
            &self.resources,
        );
        self.prefix_scan_pass
            .recreate_bind_group(&self.device, &self.resources);
        self.duplicate_pass
            .recreate_bind_group(&self.device, &self.resources);
        self.radix_sort_pass
            .recreate_bind_group(&self.device, &self.resources);
        self.tile_range_pass
            .recreate_bind_group(&self.device, &self.resources);
        self.tile_render_pass.recreate_bind_group(
            &self.device,
            &self.scene_uniform_buffer,
            &self.render_texture_view,
            &self.resources,
        );
        self.axis_pass
            .recreate_bind_group(&self.device, &self.scene_uniform_buffer);
        self.scene_uniform
            .update_gaussian_count(self.resources.gaussian_count);
        self.queue.write_buffer(
            &self.scene_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.scene_uniform]),
        );
        self.scene_type = new_scene_type;
        self.scene_dirty = true;
        Ok(())
    }
}

pub struct App {
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<UserEvent>>,
    state: Option<AppState>,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<UserEvent>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
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
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::DroppedFile(path) => {
                let format = assets::format::FileFormat::from_path(&path);
                let Some(format) = format else {
                    log::error!("unsupported file format: {path:?}");
                    return;
                };
                match assets::loader::load_gaussians_from_path(format, &path)
                    .and_then(|gaussians| state.replace_gaussians(gaussians))
                {
                    Ok(()) => {
                        log::info!("loaded dropped file: {path:?}");
                        state.window.request_redraw();
                    }
                    Err(e) => {
                        log::error!("{e:?}: {path:?}");
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if state.is_occluded || !state.is_focused {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let _ = state.device.poll(wgpu::PollType::Poll);
                    }
                    return;
                }

                #[cfg(target_arch = "wasm32")]
                if !state.can_submit_frame() {
                    state
                        .wants_redraw_after_gpu_done
                        .store(true, Ordering::Relaxed);
                    return;
                }

                state.update();

                match state.render() {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("{e}");
                        event_loop.exit();
                    }
                }

                if state.should_request_redraw() {
                    #[cfg(target_arch = "wasm32")]
                    state.request_redraw_or_defer();

                    #[cfg(not(target_arch = "wasm32"))]
                    state.window.request_redraw();
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
                    state.window.request_redraw();
                } else {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let _ = state.device.poll(wgpu::PollType::Poll);
                    }
                }
            }

            WindowEvent::Occluded(occluded) => {
                state.is_occluded = occluded;

                if !occluded {
                    state.last_update_time = Instant::now();
                    state.window.request_redraw();
                } else {
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        let _ = state.device.poll(wgpu::PollType::Poll);
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
                    state.window.request_redraw();
                    state.resize(
                        state.window.inner_size().width,
                        state.window.inner_size().height,
                    );
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

                match assets::loader::load_gaussians_from_bytes(format, &bytes)
                    .and_then(|gaussians| state.replace_gaussians(gaussians))
                {
                    Ok(()) => {
                        log::info!("loaded dropped file: {file_name}");
                        state.window.request_redraw();
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
