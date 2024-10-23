import wandb
import torch


def upload_ckpt(ckpt_dir, ckpt_name):
    ckpt_file = f"{ckpt_dir}/{ckpt_name}"
    print(f"upload ckpt: {ckpt_file}")

    artifact = wandb.Artifact("model_weights", type="model")
    artifact.add_file(ckpt_file)

    with wandb.init(project="FCTL-stage2", entity="Jankiny", job_type="upload", anonymous="allow"):
        wandb.log_artifact(artifact)
    # checkpoint = torch.load(ckpt_file)
    # metadata = {}
    # for key in checkpoint.keys():
    #     if key not in ['state_dict', 'loops', 'optimizer_states']:
    #         metadata[key] = checkpoint[key]
    # wandb.log({"checkpoint": wandb.Artifact("checkpoint", type="model", metadata=metadata)})


if __name__ == '__main__':
    wandb.require("core")

    exp_name = "common_vitb16_TinyImageNet_split5_2_exp"
    upload_ckpt(f"runs/report/{exp_name}/ckpts", "best.ckpt")

    wandb.finish()