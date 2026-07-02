pub mod bindings;
pub mod global_sort_pipeline_resources;

use self::bindings::{
    build_indirect_args::BuildIndirectArgsBindings, preprocess::PreprocessBindings,
    radix_sort::RadixSortBindings, render::RenderBindings,
};
use self::global_sort_pipeline_resources::GlobalSortPipelineResources;
use crate::resources::{
    gaussians::{GaussianResource, Gaussians},
    scene::SceneResource,
};

use crate::camera::Camera;
use crate::gpu::context::GpuContext;
use crate::scene::Scene;

pub const RADIX_SORT_PASSES: u64 = 4;

pub struct GlobalSortResources {
    pub gpu_resources: GlobalSortGpuResources,
    pub bindings: GlobalSortBindings,
}

impl GlobalSortResources {
    pub fn new(gpu: &GpuContext, gaussians: &Gaussians, camera: &Camera) -> Self {
        let gpu_resources = GlobalSortGpuResources::new(gpu, gaussians, camera);
        let bindings = GlobalSortBindings::new(gpu, &gpu_resources);
        Self {
            gpu_resources,
            bindings,
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.gpu_resources.resize(gpu, scene, camera);
    }

    pub fn replace_gaussians(&mut self, gpu: &GpuContext, gaussians: &Gaussians) {
        self.gpu_resources.replace_gaussians(gpu, gaussians);
        self.bindings.recreate(gpu, &self.gpu_resources);
    }
}

pub struct GlobalSortGpuResources {
    pub pipeline: GlobalSortPipelineResources,
    pub gaussian: GaussianResource,
    pub scene: SceneResource,
}

impl GlobalSortGpuResources {
    pub fn new(gpu: &GpuContext, gaussians: &Gaussians, camera: &Camera) -> Self {
        let gaussian_count = gaussians.len() as u32;
        Self {
            pipeline: GlobalSortPipelineResources::new(gpu, gaussian_count),
            gaussian: GaussianResource::new(gpu, gaussians),
            scene: SceneResource::new(gpu, camera),
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.scene.update(gpu, scene, camera);
    }

    pub fn replace_gaussians(&mut self, gpu: &GpuContext, gaussians: &Gaussians) {
        let gaussian_count = gaussians.len() as u32;
        self.pipeline = GlobalSortPipelineResources::new(gpu, gaussian_count);
        self.gaussian = GaussianResource::new(gpu, gaussians);
    }
}

pub struct GlobalSortBindings {
    pub preprocess: PreprocessBindings,
    pub build_indirect_args: BuildIndirectArgsBindings,
    pub radix_sort: RadixSortBindings,
    pub render: RenderBindings,
}

impl GlobalSortBindings {
    fn new(gpu: &GpuContext, gpu_resources: &GlobalSortGpuResources) -> Self {
        Self {
            preprocess: PreprocessBindings::new(gpu, gpu_resources),
            build_indirect_args: BuildIndirectArgsBindings::new(gpu, gpu_resources),
            radix_sort: RadixSortBindings::new(gpu, gpu_resources),
            render: RenderBindings::new(gpu, gpu_resources),
        }
    }

    fn recreate(&mut self, gpu: &GpuContext, gpu_resources: &GlobalSortGpuResources) {
        self.preprocess.recreate(gpu, gpu_resources);
        self.build_indirect_args.recreate(gpu, gpu_resources);
        self.radix_sort.recreate(gpu, gpu_resources);
        self.render.recreate(gpu, gpu_resources);
    }
}
