use super::passes::{
    build_indirect_args::BuildIndirectArgsPass, preprocess::PreprocessPass,
    radix_sort::RadixSortPass, render::RenderPass,
};
use super::resources::GlobalSortResources;
use crate::camera::Camera;
use crate::gpu::context::GpuContext;
use crate::renderer::axis::AxisPass;
use crate::resources::gaussians::Gaussians;
use crate::scene::{Scene, SceneType};

pub struct GlobalSortRenderer {
    resources: GlobalSortResources,

    preprocess_pass: PreprocessPass,
    build_indirect_args_pass: BuildIndirectArgsPass,
    radix_sort_pass: RadixSortPass,
    render_pass: RenderPass,

    axis_pass: AxisPass,

    scene_type: SceneType,

    dirty: bool,
}

impl GlobalSortRenderer {
    pub fn new(
        gpu: &GpuContext,
        gaussians: &Gaussians,
        camera: &Camera,
        scene_type: SceneType,
    ) -> Self {
        let resources = GlobalSortResources::new(gpu, gaussians, camera);

        let preprocess_pass = PreprocessPass::new(gpu, &resources.bindings.preprocess, scene_type);
        let build_indirect_args_pass =
            BuildIndirectArgsPass::new(gpu, &resources.bindings.build_indirect_args);
        let radix_sort_pass = RadixSortPass::new(gpu, &resources.bindings.radix_sort);
        let render_pass = RenderPass::new(gpu, &resources.bindings.render);
        let axis_pass = AxisPass::new(gpu, &resources.gpu_resources.scene);

        Self {
            resources,
            preprocess_pass,
            build_indirect_args_pass,
            radix_sort_pass,
            render_pass,
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
            self.build_indirect_args_pass
                .encode(encoder, &self.resources.bindings.build_indirect_args);
            self.radix_sort_pass.encode(
                encoder,
                &self.resources.gpu_resources.pipeline,
                &self.resources.bindings.radix_sort,
            );
            self.dirty = false;
        }
        self.render_pass.encode(
            encoder,
            &self.resources.gpu_resources.pipeline,
            &self.resources.bindings.render,
            output_view,
        );

        self.axis_pass.encode(encoder, output_view);
    }

    pub fn resize(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.resources.resize(gpu, scene, camera);
        self.dirty = true;
    }

    pub fn update_scene_uniform(&mut self, gpu: &GpuContext, scene: &Scene, camera: &Camera) {
        self.resources
            .gpu_resources
            .scene
            .update(gpu, scene, camera);
        self.dirty = true;
    }

    pub fn replace_gaussians(
        &mut self,
        gpu: &GpuContext,
        gaussians: &Gaussians,
        scene_type: SceneType,
    ) {
        self.resources.replace_gaussians(gpu, gaussians);
        if self.scene_type != scene_type {
            self.preprocess_pass =
                PreprocessPass::new(gpu, &self.resources.bindings.preprocess, scene_type);
        }
        self.scene_type = scene_type;
        self.dirty = true;
    }
}
