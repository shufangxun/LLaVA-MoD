import os
from .clip_encoder import CLIPVisionTower
from .clips2_encoder import CLIPVisionTowerS2
import transformers

a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower


# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    # is_absolute_path_exists = os.path.exists(image_tower)
    use_s2 = getattr(image_tower_cfg, 's2', False)
    print("#### image tower config")
    print(image_tower_cfg)
    print("##### use clips2: ", use_s2)
    # import pdb
    # pdb.set_trace()

    if "openai" in image_tower or "laion" in image_tower:
        if use_s2:
            return CLIPVisionTowerS2(image_tower, args=image_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if "google" in image_tower:
        return SiglipVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if "LanguageBind_Image" in image_tower:
        return
    #     return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')


def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return
        # return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
