use crate::camera::Camera;
use crate::gpu::context::GpuContext;
use crate::redraw_scheduler::RedrawScheduler;
use crate::resources::gaussians::Gaussians;
use crate::scene::{Scene, SceneType};

#[cfg(not(feature = "tile-renderer"))]
pub use super::global_sort::renderer::GlobalSortRenderer as ActiveRendererBackend;
#[cfg(feature = "tile-renderer")]
pub use super::tile::renderer::TileRenderer as ActiveRendererBackend;

#[derive(PartialEq, Eq)]
pub enum RenderStatus {
    Submitted,
    Skipped,
}

pub struct Renderer {
    backend: ActiveRendererBackend,
}

impl Renderer {
    pub fn new(
        gpu: &GpuContext,
        camera: &Camera,
        gaussians: &Gaussians,
        scene_type: SceneType,
    ) -> Self {
        let backend = ActiveRendererBackend::new(gpu, gaussians, camera, scene_type);

        Self { backend }
    }

    pub fn update(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.backend.update_scene_uniform(gpu, scene, camera);
    }

    pub fn render(
        &mut self,
        gpu: &GpuContext,
        redraw_scheduler: &RedrawScheduler,
    ) -> anyhow::Result<RenderStatus> {
        if !gpu.is_surface_configured() {
            redraw_scheduler.cancel_frame();
            return Ok(RenderStatus::Skipped);
        }

        let output = match gpu.get_current_texture() {
            Ok(Some(output)) => output,

            Ok(None) => {
                redraw_scheduler.cancel_frame();
                return Ok(RenderStatus::Skipped);
            }

            Err(error) => {
                redraw_scheduler.cancel_frame();
                return Err(error);
            }
        };

        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu.create_command_encoder("Render Encoder");

        self.backend.render(&mut encoder, &output_view);

        redraw_scheduler.submit_frame(gpu, encoder.finish());

        output.present();

        Ok(RenderStatus::Submitted)
    }

    pub fn resize(&mut self, gpu: &mut GpuContext, scene: &Scene, camera: &Camera) {
        self.backend.resize(gpu, scene, camera);
    }

    pub fn replace_gaussians(
        &mut self,
        gpu: &mut GpuContext,
        gaussians: &Gaussians,
        scene_type: SceneType,
    ) {
        self.backend.replace_gaussians(gpu, gaussians, scene_type);
    }
}
