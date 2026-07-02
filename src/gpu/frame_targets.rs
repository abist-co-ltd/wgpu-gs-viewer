use crate::gpu::context::GpuContext;

pub struct FrameTargets {
    render_texture: wgpu::Texture,
    pub render_texture_view: wgpu::TextureView,
    pub render_texture_sampler: wgpu::Sampler,
}

impl FrameTargets {
    pub fn new(gpu: &GpuContext) -> Self {
        let surface_size = gpu.surface_size();

        let width = surface_size.width.max(1);
        let height = surface_size.height.max(1);

        let (render_texture, render_texture_view) = Self::create_render_texture(gpu, width, height);
        let render_texture_sampler = gpu.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Render Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            render_texture,
            render_texture_view,
            render_texture_sampler,
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        let (render_texture, render_texture_view) = Self::create_render_texture(gpu, width, height);

        self.render_texture = render_texture;
        self.render_texture_view = render_texture_view;
    }

    fn create_render_texture(
        gpu: &GpuContext,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = gpu.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        (texture, view)
    }
}
