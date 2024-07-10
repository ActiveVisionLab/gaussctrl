# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model, only needed in order to not save InstructPix2Pix checkpoints
"""
from dataclasses import dataclass, field
from typing import Type, Tuple, Dict
import functools
import time
from rich import box, style
from rich.panel import Panel
from rich.table import Table

import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.viewer.server.viewer_elements import ViewerButton
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str

@dataclass
class GaussEditTrainerConfig(TrainerConfig):
    """Configuration for the InstructNeRF2NeRFTrainer."""
    _target: Type = field(default_factory=lambda: GaussEditTrainer)
    steps_per_save: int = 500
    """Number of steps between saves."""
    # dataset_update_num: int = 2


class GaussEditTrainer(Trainer):
    """Trainer for InstructNeRF2NeRF"""

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:

        super().__init__(config, local_rank, world_size)
        # reset button
        self.reset_button = ViewerButton(name="Reset Button", cb_hook=self.reset_callback)

    def reset_callback(self, handle: ViewerButton) -> None:
        """Reset the model to the original checkpoint"""
        
        # load checkpoint
        self._load_checkpoint()

        # reset dataset
        self.config.pipeline.datamanager.image_batch['image'] = self.config.pipeline.datamanager.original_image_batch['image'].clone()
        self.config.pipeline.datamanager.image_batch['image_idx'] = self.config.pipeline.datamanager.original_image_batch['image_idx'].clone()

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "ip2p." not in k}
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        
        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )

        # self.pipeline.sample_traj()
        
        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            for COUNT in range(self.pipeline.config.dataset_update_num):
                num_iterations = self.pipeline.config.render_rate
                step = 0
                for step in range(self._start_step, self._start_step + num_iterations):
                    while self.training_state == "paused":
                        time.sleep(0.01)
                    with self.train_lock:
                        with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                            self.pipeline.train()

                            # training callbacks before the training iteration
                            for callback in self.callbacks:
                                callback.run_callback_at_location(
                                    step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                                )

                            # time the forward pass
                            loss, loss_dict, metrics_dict = self.train_iteration(step, COUNT)

                            # training callbacks after the training iteration
                            for callback in self.callbacks:
                                callback.run_callback_at_location(
                                    step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                                )

                    self._update_viewer_state(step)

                    # a batch of train rays
                    if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                        writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                        writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                        writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                        # The actual memory allocated by Pytorch. This is likely less than the amount
                        # shown in nvidia-smi since some unused memory can be held by the caching
                        # allocator and some context needs to be created on GPU. See Memory management
                        # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                        # for more details about GPU memory management.
                        writer.put_scalar(
                            name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                        )

                    # Do not perform evaluation if there are no validation images
                    if self.pipeline.datamanager.eval_dataset:
                        self.eval_iteration(step)

                    if step_check(step, self.config.steps_per_save):
                        self.save_checkpoint(step)

                    writer.write_out_storage()

                if COUNT < self.pipeline.config.dataset_update_num - 1:
                    if ((step+1) % self.pipeline.config.render_rate == 0):
                        CONSOLE.print("Reloading initial checkpoints",style="bold yellow")
                        self._load_checkpoint()
                        # breakpoint()
                        CONSOLE.print("Done reloading initial checkpoints",style="bold yellow")

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    @profiler.time_function
    def train_iteration(self, step: int, count: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str
        
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step, count=count)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore