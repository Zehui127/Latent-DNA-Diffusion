import os
import torch
import datetime
from accelerate import Accelerator
from ..models.vanilla_vae.vae import VanillaVAE
from ..models.unet.unet_model import *
from ..datasets.lightning_loader import *
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from pathlib import Path
from diffusers import DDPMScheduler
from ..utils.tensor_to_dna import tensor_to_dna
from ..models.unet.custom_scheduler_uncondition import DDPMDNAPipeline
from ..utils.evaluator import write_to_fasta
from tqdm import tqdm


def train_loop(config):
    config = config
    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir),
        cpu=False
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(TIME)

    data_module = DataModule(config.loader_config)

    accelerator.wait_for_everyone()
    data_module.setup()
    train_loader, val_loader = data_module.train_dataloader(), data_module.val_dataloader()

    print("len train loader", len(train_loader))
    config.device = accelerator.device

    vae = VanillaVAE(in_channels = 1,
        latent_dim = 256,
        hidden_dims = [8, 16, 32],
        seq2img_num_layers = 4,
        seq2img_img_width = 128,
        kld_weight = 0.00001)

    vae.to(config.device)

    #load vae params
    state_dict=torch.load(config.vae_path, map_location= config.device)
    vae.load_state_dict(state_dict)

    #unet model
    model = UNet2DModel(sample_size=config.sample_size, in_channels= config.in_channels, out_channels= config.out_channels, \
                      layers_per_block=config.layers_per_block, block_out_channels=config.block_out_channels, \
                        down_block_types=config.down_block_types, up_block_types=config.up_block_types)
    model.to(config.device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000,clip_sample=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_loader) * config.epoch),
    )

    # Prepare everything
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    ############################
    # Training
    ############################
    best_val_loss = float('inf')
    global_step = 0
    # Now you train the model
    for epoch in range(config.epoch):
        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                clean_data = batch.to(config.device)
                with torch.no_grad():
                    clean_images = vae.encode(input = clean_data)
                    clean_images = clean_images.sample()
                clean_images = clean_images.to( config.device)
                # Sample noise to add to the images
                noise = torch.randn((clean_images.shape))
                noise = noise.to(config.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bs,)
                ).long()
                timesteps = timesteps.to(config.device)

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps,return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)

                accelerator.backward(loss, retain_graph=True)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            logs = {"loss": loss.detach().item(),"lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step += 1
        if accelerator.is_local_main_process:
            model.eval()
            with torch.no_grad():
                loss_li = []
                for batch_idx, data in enumerate(val_loader):
                    clean_data = data.to(config.device)
                    clean_images = vae.encode(input = clean_data) #TODO need to verify
                    clean_images = clean_images.sample()
                    clean_images = clean_images.to(config.device)

                    # Sample noise to add to the images
                    noise = torch.randn((clean_images.shape))
                    noise = noise.to(config.device)
                    bs = clean_images.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bs,)
                    ).long()
                    timesteps = timesteps.to(config.device)

                    # forward
                    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0] #TODO need to verify
                    v_loss = F.mse_loss(noise_pred, noise)
                    loss_li.append(v_loss.mean().item())


                val_loss = sum(loss_li)/len(loss_li)
                accelerator.log({"valid_loss": val_loss}, step=global_step)
                print(f"Epoch: {epoch} / {config.epoch}, \
                        Train Loss: {loss.mean().item():.4f}, \
                        Val Loss: {val_loss:.4f}, \
                        "
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+"_best_unet_model.pth"))
            if epoch % 200 == 0 and epoch != 0:
                torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+f"_score_func_{epoch}.pth"))
                if epoch == 600:
                    exit()
                with torch.no_grad():
                    batch_size = 500
                    total_samples = 2000
                    pipe = DDPMDNAPipeline(unet = model.module, scheduler = noise_scheduler)
                    pipe = pipe.to(config.device)
                    all_samples = []
                    # create samples by batch
                    for i in range(total_samples//batch_size):
                        DNA_onehot = pipe(output_type = np.array, return_dict= False,
                                        batch_size=batch_size,vae=vae)
                        DNA_onehot = DNA_onehot[0]
                        seq = [tensor_to_dna(s) for s in DNA_onehot]
                        all_samples.extend(seq)
                    # last batch
                    if total_samples%batch_size != 0:
                        batch_size = total_samples%batch_size
                        DNA_onehot = pipe(output_type = np.array, return_dict= False,
                                            batch_size=batch_size,vae=vae)
                        DNA_onehot = DNA_onehot[0]
                        seq = [tensor_to_dna(s) for s in DNA_onehot]
                        all_samples.extend(seq)
                        # write to fasta
                    write_to_fasta(seq, os.path.join(config.save_path, str(epoch)+"_generated_DNA.fasta"))
        accelerator.wait_for_everyone()
    torch.save(model.module.state_dict(), os.path.join(config.save_path, TIME+"_last_unet_model.pth"))
    print(">>> Training finished.")
    accelerator.wait_for_everyone()
    accelerator.end_training()

def evaluate(config,vae_ckpt,unet_ckpt):
    unet_version = unet_ckpt.split("/")[-1]
    sequence_num = config.sequence_num
    config = config
    TIME = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    accelerator = Accelerator(
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir),
        cpu=False
    )
    # only run on the main process
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers(TIME)
        config.device = accelerator.device
        ## load VAE
        vae = VanillaVAE(in_channels = 1,
            latent_dim = 256,
            hidden_dims = [8, 16, 32],
            seq2img_num_layers = 4,
            seq2img_img_width = 128,
            kld_weight = 0.00001)
        vae.to(config.device)
        #load vae params
        state_dict = torch.load(vae_ckpt, map_location= config.device)
        vae.load_state_dict(state_dict)

        vae.eval()
        ## load UNet
        model = UNet2DModel(sample_size=config.sample_size, in_channels= config.in_channels, out_channels= config.out_channels, \
                      layers_per_block=config.layers_per_block, block_out_channels=config.block_out_channels, \
                        down_block_types=config.down_block_types, up_block_types=config.up_block_types)
        model.to(config.device)
        # load UNet params
        state_dict = torch.load(unet_ckpt, map_location= config.device)
        model.load_state_dict(state_dict)
        model.eval()

        noise_scheduler = DDPMScheduler(num_train_timesteps=1000,clip_sample=False)

        # define pipeline:
        pipe = DDPMDNAPipeline(unet = model, scheduler = noise_scheduler)
        pipe = pipe.to(config.device)

        # generate in batches:
        batch_size = config.batch_size
        total_samples = sequence_num
        for i in tqdm(range(total_samples//batch_size)):
            DNA_onehot = pipe(output_type = np.array, return_dict= False, batch_size=batch_size,vae=vae)
            DNA_onehot = DNA_onehot[0]
            print(DNA_onehot.shape)
            dna = [tensor_to_dna(ele) for ele in DNA_onehot]
            write_to_fasta(dna, os.path.join(config.save_path, f"{TIME}_{unet_version}.fasta"))
        # last batch
        if total_samples%batch_size != 0:
            batch_size = total_samples%batch_size
            DNA_onehot = pipe(output_type = np.array, return_dict= False, batch_size=batch_size,vae=vae)
            DNA_onehot = DNA_onehot[0]
            dna = [tensor_to_dna(ele) for ele in DNA_onehot]
            write_to_fasta(dna, os.path.join(config.save_path, f"{TIME}_{unet_version}.fasta"))
        print(f">>> Generate {sequence_num} DNA sequecnes")
