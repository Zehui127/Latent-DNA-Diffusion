import os
import torch
import datetime
import numpy as np
from accelerate import Accelerator
from accelerate.utils import LoggerType
from ..models.vanilla_vae.vae import VanillaVAE
from ..datasets.reference_loader import *
from ..datasets.lightning_loader import *
from ..utils.metrics import *
# from ..datasets.epd_loader import *
from ..utils.types_ import *
from ..utils.beta_scheduler import StepBetaScheduler
import math
from ..utils.tensor_to_dna import tensor_to_dna
from tqdm.auto import tqdm
from accelerate.utils import DistributedDataParallelKwargs


def train(config, cpt_name="", cpt_numer=0):
    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with=LoggerType.TENSORBOARD, project_dir="logs",kwargs_handlers=[kwargs])
    print(TIME)
    data_module = DataModule(config.loader_config)
    accelerator.wait_for_everyone()
    data_module.setup()
    train_loader, val_loader = data_module.train_dataloader(), data_module.val_dataloader()
    print("Current device:", accelerator.device)

    model = VanillaVAE(in_channels=config.in_channel, latent_dim=config.latent_dim, hidden_dims=config.hidden_dims, \
                       seq2img_img_width=config.seq2img_img_width, seq2img_num_layers=config.seq2img_num_layers, layer_per_block=config.layer_per_block)
    # model.to(accelerator.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")
    if cpt_name != "":
        if not cpt_name.endswith(".pth"):
            cpt_name+=".pth"
        d = torch.load(os.path.join(config.save_path, cpt_name), map_location=accelerator.device)
        model.load_state_dict(d)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model, optimizer, train_data_loader, val_data_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    scheduler = StepBetaScheduler(config.epoch)
    accelerator.init_trackers(TIME)
    accelerator.wait_for_everyone()
    print("=== Training...")
    best_val_acc = 0
    global_step = 0
    for epoch in range(cpt_numer, config.epoch):
        model.train()
        kld_weight = scheduler.step()
        for batch_idx, data in enumerate(train_data_loader):
            # data = data.to(accelerator.device)
            recon_batch, dist = model(data)
            loss,_,_ = model.module.loss_function(recon_batch, data, dist, kld_weight=kld_weight)

            accelerator.backward(loss.mean())
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 2000 == 0 and accelerator.is_local_main_process:
                t_acc = batch_accuracy(recon_batch, data)
                # t_align = get_align_dist(recon_batch, data)
                print(f"Epoch: {epoch} / {config.epoch} : {batch_idx}, \
                        Train Loss: {loss.mean().item():.4f} \
                        Train Accuracy: {t_acc:.4f} \
                        Time: {str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))}", \
                        end="\r")
                # Train Align: {t_align} \
                accelerator.log({"train_acc": t_acc}, step=global_step)
                # "train_aln": t_align[0]

            accelerator.log({"train_loss": loss.mean().item(), "log_train_loss": math.log(loss.mean().item()), "kl_component": dist.kl(), "lr_schedule": optimizer.param_groups[0]['lr']}, step=global_step)
            global_step+=1


        model.eval()
        with torch.no_grad():
            loss_li = []
            acc_li = []
            rec_li = []
            kl_li = []
            # aln_li = []
            for batch_idx, data in enumerate(val_data_loader):
                # data = data.to(accelerator.device)
                recon_batch, dist = model(data)
                v_loss,rec_loss,kl_loss = model.module.loss_function(recon_batch, data, dist, kld_weight=kld_weight)
                loss_li.append(v_loss.mean().item())
                ###
                rec_li.append(rec_loss.mean().item())
                kl_li.append(kl_loss.mean().item())
                v_acc = batch_accuracy(recon_batch, data)
                acc_li.append(v_acc)
                # v_align = get_align_dist(recon_batch, data)
                # aln_li.append(v_align[1])
                if batch_idx == 0 and accelerator.is_local_main_process:
                    print(tensor_to_dna(recon_batch[0])[:100])
                    print(tensor_to_dna(data[0])[:100])

            val_loss = sum(loss_li)/len(loss_li)
            val_acc = sum(acc_li)/len(acc_li)
            ###
            val_loss_rec = sum(rec_li)/len(rec_li)
            val_loss_kl = sum(kl_li)/len(kl_li)
            # val_aln = sum(aln_li)/len(aln_li)
            accelerator.log({"valid_loss": val_loss, "log_valid_loss": math.log(val_loss), "valid_acc": val_acc}, step=global_step)
            # accelerator.log({"valid_aln": val_aln})
            print(f"Epoch: {epoch} / {config.epoch}, \
                    Val Loss: {val_loss:.4f}/ {val_loss_rec:.4f}/ {val_loss_kl:.4f}, \
                    Val Accuracy: {val_acc:.4f} \
                    Weigth: {kld_weight:.8f}\
                    "
                    # Val Align: {val_aln:.4f} \
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+"baseline_best_model.pth"))
    print(">>> Training finished. Best epoch:", best_epoch)
    print(f"Best Validation Acc {best_val_acc}")
    torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+"baseline_checkpoint.pth"))
    accelerator.wait_for_everyone()
    accelerator.end_training()
