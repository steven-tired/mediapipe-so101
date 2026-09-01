# Phase C checkpoint inventory

Updated 2026-08-24. These checkpoints are retained for matched physical
evaluation; a larger training step is not assumed to be better.

## Hugging Face finals

All three repositories are private and contain only the final
`pretrained_model` files.

| Policy | Effective step | Repository |
| --- | ---: | --- |
| ACT | 80k | `stevenzenith/act_carton_phase_c_80k` |
| SmolVLA | 80k | `stevenzenith/smolvla_carton_phase_c_80k` |
| Diffusion | 90k | `stevenzenith/diffusion_carton_phase_c_90k` |

## Local candidates

### ACT

Base directory:

`phase_c_act/outputs/act_phase_c_full_20260822_131930/checkpoints`

Retained steps: 5k through 80k at 5k intervals. Use each
`<step>/pretrained_model` directory for deployment.

### SmolVLA

The extension run starts from the original 20k model, so extension checkpoint
names are not effective total steps.

| Effective step | Local checkpoint |
| ---: | --- |
| 20k | `phase_c_smolvla_checkpoints/phase_c_smolvla/outputs/smolvla_phase_c_20k_20260822_185909/checkpoints/020000/pretrained_model` |
| 30k | `phase_c_smolvla_checkpoints/phase_c_smolvla_extend/outputs/smolvla_phase_c_extend_20k_to_80k_20260823_022644/checkpoints/010000/pretrained_model` |
| 40k | `phase_c_smolvla_checkpoints/phase_c_smolvla_extend/outputs/smolvla_phase_c_extend_20k_to_80k_20260823_022644/checkpoints/020000/pretrained_model` |
| 50k | `phase_c_smolvla_checkpoints/phase_c_smolvla_extend/outputs/smolvla_phase_c_extend_20k_to_80k_20260823_022644/checkpoints/030000/pretrained_model` |
| 60k | `phase_c_smolvla_checkpoints/phase_c_smolvla_extend/outputs/smolvla_phase_c_extend_20k_to_80k_20260823_022644/checkpoints/040000/pretrained_model` |
| 70k | `phase_c_smolvla_checkpoints/phase_c_smolvla_extend/outputs/smolvla_phase_c_extend_20k_to_80k_20260823_022644/checkpoints/050000/pretrained_model` |
| 80k | `phase_c_smolvla_checkpoints/phase_c_smolvla_extend/outputs/smolvla_phase_c_extend_20k_to_80k_20260823_022644/checkpoints/060000/pretrained_model` |

### Diffusion

Base directory:

`phase_c_diffusion_checkpoints/phase_c_diffusion/outputs/diffusion_phase_c_90k_20260823_023729/checkpoints`

Retained steps: 20k, 40k, 60k, 80k, and 90k. Use each
`<step>/pretrained_model` directory for deployment.

## Verification boundary

Every retained checkpoint has the required config, model, preprocessor, and
postprocessor files, and every `model.safetensors` index was opened
successfully. The three final policies were instantiated from both their local
directories and their private Hub repositories. This proves artifact and load
compatibility only; physical behavior still requires matched robot rollout
evaluation.
