pub mod bindings;
pub mod tile_pipeline_resources;

use self::bindings::{
    duplicate::DuplicateBindings, prefix_scan::PrefixScanBindings, preprocess::PreprocessBindings,
    radix_sort::RadixSortBindings, tile_range::TileRangeBindings, tile_render::TileRenderBindings,
};
use self::tile_pipeline_resources::TilePipelineResources;
use crate::renderer::screen_blit::ScreenBlitBindings;
use crate::resources::{
    gaussians::{GaussianResource, Gaussians},
    scene::SceneResource,
};

use crate::camera::Camera;
use crate::gpu::{context::GpuContext, frame_targets::FrameTargets};
use crate::scene::Scene;

pub const PREFIX_DISPATCH_ARGS_COUNT: u64 = 5;
pub const RADIX_SORT_PASSES: u64 = 8;

pub struct TileResources {
    pub gpu_resources: TileGpuResources,
    pub bindings: TileBindings,
}

impl TileResources {
    pub fn new(
        gpu: &GpuContext,
        frame_targets: &FrameTargets,
        gaussians: &Gaussians,
        camera: &Camera,
    ) -> Self {
        let gpu_resources = TileGpuResources::new(gpu, gaussians, camera);
        let bindings = TileBindings::new(gpu, frame_targets, &gpu_resources);
        Self {
            gpu_resources,
            bindings,
        }
    }

    pub fn resize(
        &mut self,
        gpu: &GpuContext,
        frame_targets: &FrameTargets,
        scene: &Scene,
        camera: &Camera,
    ) {
        self.gpu_resources.resize(gpu, scene, camera);
        self.bindings
            .recreate(gpu, frame_targets, &self.gpu_resources);
    }

    pub fn replace_gaussians(
        &mut self,
        gpu: &GpuContext,
        frame_targets: &FrameTargets,
        gaussians: &Gaussians,
    ) {
        self.gpu_resources.replace_gaussians(gpu, gaussians);
        self.bindings
            .recreate(gpu, frame_targets, &self.gpu_resources);
    }
}

pub struct TileGpuResources {
    pub pipeline: TilePipelineResources,
    pub gaussian: GaussianResource,
    pub scene: SceneResource,
}

impl TileGpuResources {
    pub fn new(gpu: &GpuContext, gaussians: &Gaussians, camera: &Camera) -> Self {
        let gaussian_count = gaussians.len() as u32;
        Self {
            pipeline: TilePipelineResources::new(gpu, gaussian_count),
            gaussian: GaussianResource::new(gpu, gaussians),
            scene: SceneResource::new(gpu, camera),
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.pipeline.resize(gpu);
        self.scene.update(gpu, scene, camera);
    }

    pub fn replace_gaussians(&mut self, gpu: &GpuContext, gaussians: &Gaussians) {
        let gaussian_count = gaussians.len() as u32;
        self.pipeline = TilePipelineResources::new(gpu, gaussian_count);
        self.gaussian = GaussianResource::new(gpu, gaussians);
    }
}

pub struct TileBindings {
    pub preprocess: PreprocessBindings,
    pub prefix_scan: PrefixScanBindings,
    pub duplicate: DuplicateBindings,
    pub radix_sort: RadixSortBindings,
    pub tile_range: TileRangeBindings,
    pub tile_render: TileRenderBindings,
    pub screen_blit: ScreenBlitBindings,
}

impl TileBindings {
    fn new(
        gpu: &GpuContext,
        frame_targets: &FrameTargets,
        gpu_resources: &TileGpuResources,
    ) -> Self {
        Self {
            preprocess: PreprocessBindings::new(gpu, gpu_resources),
            prefix_scan: PrefixScanBindings::new(gpu, gpu_resources),
            duplicate: DuplicateBindings::new(gpu, gpu_resources),
            radix_sort: RadixSortBindings::new(gpu, gpu_resources),
            tile_range: TileRangeBindings::new(gpu, gpu_resources),
            tile_render: TileRenderBindings::new(gpu, frame_targets, gpu_resources),
            screen_blit: ScreenBlitBindings::new(gpu, frame_targets),
        }
    }

    fn recreate(
        &mut self,
        gpu: &GpuContext,
        frame_targets: &FrameTargets,
        gpu_resources: &TileGpuResources,
    ) {
        self.preprocess.recreate(gpu, gpu_resources);
        self.prefix_scan.recreate(gpu, gpu_resources);
        self.duplicate.recreate(gpu, gpu_resources);
        self.radix_sort.recreate(gpu, gpu_resources);
        self.tile_range.recreate(gpu, gpu_resources);
        self.tile_render.recreate(gpu, frame_targets, gpu_resources);
        self.screen_blit.recreate(gpu, frame_targets);
    }
}
