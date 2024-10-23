import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from .clip_encoder import CLIPVisionTower


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, image_tower, args, delay_load=False):
        # super().__init__(image_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(image_tower, args, delay_load)

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError(
                'Package s2wrapper not found! Please install by running: \npip install git@gitlab.alibaba-inc.com:guanghao.zgh/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.image_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_tower_name)
        self.image_tower = CLIPVisionModel.from_pretrained(self.image_tower_name)
        self.image_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype),
                                              output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        # import ipdb
        # ipdb.set_trace()
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0),
                                                        img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales,
                                                     max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)