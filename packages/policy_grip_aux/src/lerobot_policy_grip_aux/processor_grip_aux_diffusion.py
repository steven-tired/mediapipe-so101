from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors


def make_grip_aux_diffusion_pre_post_processors(config, dataset_stats=None):
    return make_diffusion_pre_post_processors(config, dataset_stats=dataset_stats)
