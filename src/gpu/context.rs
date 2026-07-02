use anyhow::{Context, bail};
use std::sync::Arc;
use winit::window::Window;

pub struct GpuContext {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,
    surface_configured: bool,
    scale_factor: f64,
}

impl GpuContext {
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

        let surface = instance
            .create_surface(window.clone())
            .with_context(|| "failed to crate surface")?;

        #[cfg(not(target_arch = "wasm32"))]
        let adapter = instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .find(|adapter| adapter.is_surface_supported(&surface))
            .with_context(|| {
                format!(
                    "failed to find a GPU adapter compatible with the surface; backends: {:?}",
                    wgpu::Backends::all()
                )
            })?;

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
            .or_else(|| surface_caps.formats.first().copied())
            .context("surface does not support any texture formats")?;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);
        Ok(Self {
            surface,
            device,
            queue,
            surface_config,
            surface_configured: true,
            scale_factor: 1.0,
        })
    }

    pub fn resize_surface(&mut self, width: u32, height: u32, scale_factor: f64) -> bool {
        if width == 0 || height == 0 {
            return false;
        }

        if self.surface_config.width == width && self.surface_config.height == height {
            return false;
        }

        self.surface_config.width = width;
        self.surface_config.height = height;

        self.surface.configure(&self.device, &self.surface_config);

        self.surface_configured = true;

        self.scale_factor = scale_factor.max(1.0);

        true
    }

    pub fn create_buffer(
        &self,
        label: &str,
        size: wgpu::BufferAddress,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    pub fn create_buffer_init<T: bytemuck::Pod>(
        &self,
        label: &str,
        data: &[T],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;

        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    pub fn write_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[T],
    ) {
        self.queue
            .write_buffer(buffer, offset, bytemuck::cast_slice(data));
    }

    pub fn write_value<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, value: &T) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::bytes_of(value));
    }

    pub fn create_bind_group_layout(
        &self,
        label: &str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        self.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries,
            })
    }

    pub fn create_bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry<'_>],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries,
        })
    }

    pub fn create_pipeline_layout(
        &self,
        label: &str,
        bind_group_layouts: &[Option<&wgpu::BindGroupLayout>],
    ) -> wgpu::PipelineLayout {
        self.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts,
                immediate_size: 0,
            })
    }

    pub fn create_compute_pipeline(
        &self,
        descriptor: &wgpu::ComputePipelineDescriptor,
    ) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(descriptor)
    }

    pub fn create_render_pipeline(
        &self,
        descriptor: &wgpu::RenderPipelineDescriptor,
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(descriptor)
    }

    pub fn create_shader_module(
        &self,
        descriptor: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::ShaderModule {
        self.device.create_shader_module(descriptor)
    }

    pub fn get_current_texture(&self) -> anyhow::Result<Option<wgpu::SurfaceTexture>> {
        let surface_texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.surface.configure(&self.device, &self.surface_config);
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                bail!("surface was lost");
            }
        };

        Ok(Some(surface_texture))
    }

    pub fn create_texture(&self, descriptor: &wgpu::TextureDescriptor<'_>) -> wgpu::Texture {
        self.device.create_texture(descriptor)
    }

    pub fn create_sampler(&self, descriptor: &wgpu::SamplerDescriptor<'_>) -> wgpu::Sampler {
        self.device.create_sampler(descriptor)
    }

    pub fn is_surface_configured(&self) -> bool {
        self.surface_configured
    }

    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_config.format
    }

    pub fn surface_size(&self) -> winit::dpi::PhysicalSize<u32> {
        winit::dpi::PhysicalSize::new(self.surface_config.width, self.surface_config.height)
    }

    pub fn create_command_encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    pub fn submit(&self, command_buffer: wgpu::CommandBuffer) -> wgpu::SubmissionIndex {
        self.queue.submit([command_buffer])
    }

    pub fn on_submitted_work_done<F>(&self, callback: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.queue.on_submitted_work_done(callback);
    }

    pub fn submit_with_callback<F>(
        &self,
        command_buffer: wgpu::CommandBuffer,
        callback: F,
    ) -> wgpu::SubmissionIndex
    where
        F: FnOnce() + Send + 'static,
    {
        let submission_index = self.submit(command_buffer);
        self.on_submitted_work_done(callback);
        submission_index
    }

    pub fn poll_once(&self) -> anyhow::Result<()> {
        self.device.poll(wgpu::PollType::Poll)?;
        Ok(())
    }
}
