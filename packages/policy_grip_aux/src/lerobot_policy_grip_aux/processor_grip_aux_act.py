from lerobot.policies.act.processor_act import make_act_pre_post_processors


def make_grip_aux_act_pre_post_processors(config, dataset_stats=None):
    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
