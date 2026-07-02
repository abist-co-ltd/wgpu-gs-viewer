use super::passes::{
    duplicate::DuplicatePass, prefix_scan::PrefixScanPass, preprocess::PreprocessPass,
    radix_sort::RadixSortPass, tile_range::TileRangePass, tile_render::TileRenderPass,
};
use super::resources::TileResources;
use crate::camera::Camera;
use crate::gpu::{context::GpuContext, frame_targets::FrameTargets};
use crate::renderer::{axis::AxisPass, screen_blit::ScreenBlitPass};
use crate::resources::gaussians::Gaussians;
use crate::scene::{Scene, SceneType};

pub struct TileRenderer {
    frame_targets: FrameTargets,

    resources: TileResources,

    preprocess_pass: PreprocessPass,
    prefix_scan_pass: PrefixScanPass,
    duplicate_pass: DuplicatePass,
    radix_sort_pass: RadixSortPass,
    tile_range_pass: TileRangePass,
    tile_render_pass: TileRenderPass,
    screen_blit_pass: ScreenBlitPass,

    axis_pass: AxisPass,

    scene_type: SceneType,

    dirty: bool,
}

impl TileRenderer {
    pub fn new(
        gpu: &GpuContext,
        gaussians: &Gaussians,
        camera: &Camera,
        scene_type: SceneType,
    ) -> Self {
        let frame_targets = FrameTargets::new(gpu);

        let resources = TileResources::new(gpu, &frame_targets, gaussians, camera);

        let preprocess_pass = PreprocessPass::new(gpu, &resources.bindings.preprocess, scene_type);
        let prefix_scan_pass = PrefixScanPass::new(gpu, &resources.bindings.prefix_scan);
        let duplicate_pass = DuplicatePass::new(gpu, &resources.bindings.duplicate);
        let radix_sort_pass = RadixSortPass::new(gpu, &resources.bindings.radix_sort);
        let tile_range_pass = TileRangePass::new(gpu, &resources.bindings.tile_range);
        let tile_render_pass = TileRenderPass::new(gpu, &resources.bindings.tile_render);
        let screen_blit_pass = ScreenBlitPass::new(gpu, &resources.bindings.screen_blit);
        let axis_pass = AxisPass::new(gpu, &resources.gpu_resources.scene);

        Self {
            frame_targets,
            resources,
            preprocess_pass,
            prefix_scan_pass,
            duplicate_pass,
            radix_sort_pass,
            tile_range_pass,
            tile_render_pass,
            screen_blit_pass,
            axis_pass,
            scene_type,
            dirty: true,
        }
    }

    pub fn render(&mut self, encoder: &mut wgpu::CommandEncoder, output_view: &wgpu::TextureView) {
        if self.dirty || self.scene_type.is_dynamic() {
            self.preprocess_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.preprocess,
            );
            self.prefix_scan_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.prefix_scan,
            );
            self.duplicate_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.duplicate,
            );
            self.radix_sort_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.radix_sort,
            );
            self.tile_range_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.tile_range,
            );
            self.tile_render_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.tile_render,
            );
            self.dirty = false;
        }

        self.screen_blit_pass
            .encode(encoder, &self.resources.bindings.screen_blit, output_view);
        self.axis_pass.encode(encoder, output_view);
    }

    pub fn update_scene_uniform(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.resources
            .gpu_resources
            .scene
            .update(gpu, scene, camera);
        self.dirty = true;
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        let surface_size = gpu.surface_size();
        self.frame_targets
            .resize(gpu, surface_size.width, surface_size.height);
        self.resources
            .resize(gpu, &self.frame_targets, scene, camera);
        self.dirty = true;
    }

    pub fn replace_gaussians(
        &mut self,
        gpu: &GpuContext,
        gaussians: &Gaussians,
        scene_type: SceneType,
    ) {
        self.resources
            .replace_gaussians(gpu, &self.frame_targets, gaussians);
        if self.scene_type != scene_type {
            self.preprocess_pass =
                PreprocessPass::new(gpu, &self.resources.bindings.preprocess, scene_type);
        }
        self.scene_type = scene_type;
        self.dirty = true;
    }
}
