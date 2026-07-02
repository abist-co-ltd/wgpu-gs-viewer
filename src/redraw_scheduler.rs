use std::sync::Arc;
use winit::window::Window;

#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::gpu::context::GpuContext;

#[cfg(target_arch = "wasm32")]
const MAX_FRAMES_IN_FLIGHT: u32 = 1;

#[derive(Clone)]
pub struct RedrawScheduler {
    window: Arc<Window>,

    #[cfg(target_arch = "wasm32")]
    frames_in_flight: Arc<AtomicU32>,

    #[cfg(target_arch = "wasm32")]
    redraw_pending: Arc<AtomicBool>,
}

impl RedrawScheduler {
    pub fn new(window: Arc<Window>) -> Self {
        Self {
            window,

            #[cfg(target_arch = "wasm32")]
            frames_in_flight: Arc::new(AtomicU32::new(0)),

            #[cfg(target_arch = "wasm32")]
            redraw_pending: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn request_redraw(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            if self.can_begin_frame() {
                self.window.request_redraw();
            } else {
                self.redraw_pending.store(true, Ordering::Relaxed);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            self.window.request_redraw();
        }
    }

    pub fn try_begin_frame(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            let acquired = self
                .frames_in_flight
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    (current < MAX_FRAMES_IN_FLIGHT).then_some(current + 1)
                })
                .is_ok();

            if !acquired {
                self.redraw_pending.store(true, Ordering::Relaxed);
            }

            acquired
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            true
        }
    }

    pub fn cancel_frame(&self) {
        #[cfg(target_arch = "wasm32")]
        {
            self.finish_frame();
        }
    }

    pub fn submit_frame(&self, gpu: &GpuContext, command_buffer: wgpu::CommandBuffer) {
        #[cfg(target_arch = "wasm32")]
        {
            gpu.submit(command_buffer);
            self.on_frame_submitted(gpu);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            gpu.submit(command_buffer);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn on_frame_submitted(&self, gpu: &GpuContext) {
        let scheduler = self.clone();

        gpu.on_submitted_work_done(move || {
            scheduler.finish_frame();
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn finish_frame(&self) {
        self.frames_in_flight
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(1))
            })
            .ok();

        if self.redraw_pending.swap(false, Ordering::Relaxed) {
            self.window.request_redraw();
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn can_begin_frame(&self) -> bool {
        self.frames_in_flight.load(Ordering::Relaxed) < MAX_FRAMES_IN_FLIGHT
    }
}
