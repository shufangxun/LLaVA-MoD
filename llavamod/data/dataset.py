import os
import json
import random
from dataclasses import dataclass

from torch.utils.data import Dataset

from llavamod.utils import order_pick_k
from llavamod.data.data_utils import *

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


"""
#############################################################
############# Supervised Finetuning Dataset  ################
#############################################################
"""

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazySupervisedDataset, self).__init__()
        # ================================================
        list_data_dict = []
        for data in data_path:
            rank0_print("#### read from", data)
            data = json.load(open(data, "r"))
            rank0_print("#### len: ", len(data))
            for i in data:
                i['id'] = len(list_data_dict)
                list_data_dict.append(i)
        rank0_print("#### total len:", len(list_data_dict))
        # ================================================

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # ===========================================================================
            cur_len = cur_len if ('image' in sample or 'video' in sample) else -cur_len
            # ===========================================================================
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

            if 'image' in sources[0] and 'video' not in sources[0]:
                # rank0_print('image')
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor
                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                # print(f"total {len(self.list_data_dict[i]['image'])} now {len(image_file)}")
                fallback_image = Image.new(mode="RGB", size=(224, 224), color=(0, 0, 0))
                image = []
                for file in image_file:
                    try:
                        img = Image.open(os.path.join(image_folder, file)).convert('RGB')
                        image.append(img)
                    except Exception as e:
                        print(f"Error opening image {file}: {e}, using fallback image.")
                        image.append(fallback_image)

                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]

                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

                data_dict = preprocess(sources, self.tokenizer, has_image=True)

            elif 'image' not in sources[0] and 'video' in sources[0]:
                # rank0_print('video')
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                data_dict = preprocess(sources, self.tokenizer, has_image=True)

            elif 'image' in sources[0] and 'video' in sources[0]:

                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor

                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor

                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]

                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                video = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image

                image = video + image  # video must before image

                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                data_dict = preprocess(sources, self.tokenizer, has_image=True)
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
                data_dict = preprocess(sources, self.tokenizer, has_image=False)

            # ==========================================================================================================

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                 labels=data_dict["labels"][0])

            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # the image does not exist in the data, but the model is multimodal
                if hasattr(self.data_args.image_processor, 'crop_size'):
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
                else:
                    size = self.data_args.image_processor.size
                    data_dict['image'] = [torch.zeros(3, size['height'], size['width'])]

            return data_dict

        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # print('after Collator', batch)
        # print(input_ids, labels, input_ids.ne(self.tokenizer.pad_token_id))
        # ======================================================================================================
        # origin image, if batch_size=6: [[image], [image], [video], [image, image], [video, video], [video, image]]
        '''
            will be converted to a sequence of list, if batch size=6:
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(8, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
        '''
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]

            # adapt to multi-video or multi-image or multi-image & video
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
            batch['images'] = images

        else:
            raise ValueError(f'pretrain, {instances}')



        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


"""
##################################################
################### DPO Dataset ##################
##################################################
"""

class LazyDPODataset(Dataset):
    """Dataset for preference learning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        super(LazyDPODataset, self).__init__()
        # ================================================
        list_data_dict = []
        for data in data_path:
            rank0_print("#### read from", data)
            data = json.load(open(data, "r"))
            rank0_print("#### len: ", len(data))
            for i in data:
                i['id'] = len(list_data_dict)
                list_data_dict.append(i)
        rank0_print("#### total len:", len(list_data_dict))
        # ================================================

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # ===========================================================================
            cur_len = cur_len if ('image' in sample or 'video' in sample) else -cur_len
            # ===========================================================================
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
            image:
            {
                'id': 'GCC_train_001870201',
                'image': 'GCC_train_001870201.jpg',
                'chosen': [
                    {
                        'from': 'human',
                        'value': '<image>\nProvide a brief description of the given image.'},
                    {
                        'from': 'gpt',
                        'value': 'a cartoon illustration of a winged buffalo with an angry expression .'}
                ]
                'rejected': [
                    {
                        'from': 'human',
                        'value': '<image>\nProvide a brief description of the given image.'},
                    {
                        'from': 'gpt',
                        'value': 'a cartoon illustration of a winged buffalo with an angry expression .'}
                ]
            }
        """
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            # ======================================================================================================
            if 'image' in sources[0] and 'video' not in sources[0]:
                # rank0_print('image')
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor
                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                # print(f"total {len(self.list_data_dict[i]['image'])} now {len(image_file)}")
                fallback_image = Image.new(mode="RGB", size=(224, 224), color=(0, 0, 0))
                image = []
                for file in image_file:
                    try:
                        img = Image.open(os.path.join(image_folder, file)).convert('RGB')
                        image.append(img)
                    except IOError as e:
                        print(f"Error opening image {file}: {e}, using fallback image.")
                        image.append(fallback_image)

                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]

                chosen_sources = preprocess_multimodal(copy.deepcopy([e["chosen"] for e in sources]),
                                                       self.data_args)
                rejected_sources = preprocess_multimodal(copy.deepcopy([e["rejected"] for e in sources]),
                                                         self.data_args)
                chosen_data_dict = preprocess(chosen_sources, self.tokenizer, has_image=True)
                rejected_data_dict = preprocess(rejected_sources, self.tokenizer, has_image=True)


            elif 'image' not in sources[0] and 'video' in sources[0]:
                # rank0_print('video')
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor
                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                image = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image
                # image = [torch.randn(3, 8, 224, 224) for i in video]  # fake image
                chosen_sources = preprocess_multimodal(copy.deepcopy([e["chosen"] for e in sources]),
                                                       self.data_args)
                rejected_sources = preprocess_multimodal(copy.deepcopy([e["rejected"] for e in sources]),
                                                         self.data_args)
                chosen_data_dict = preprocess(chosen_sources, self.tokenizer, has_image=True)
                rejected_data_dict = preprocess(rejected_sources, self.tokenizer, has_image=True)

            elif 'image' in sources[0] and 'video' in sources[0]:
                video_file = self.list_data_dict[i]['video']
                video_folder = self.data_args.video_folder
                video_processor = self.data_args.video_processor

                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                image_processor = self.data_args.image_processor

                image_file = image_file if isinstance(image_file, list) else [image_file]
                image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
                image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
                if self.data_args.image_aspect_ratio == 'pad':
                    image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]

                video_file = video_file if isinstance(video_file, list) else [video_file]
                video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
                video = [os.path.join(video_folder, file) for file in video_file]
                video = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image

                image = video + image  # video must before image

                chosen_sources = preprocess_multimodal(copy.deepcopy([e["chosen"] for e in sources]),
                                                       self.data_args)
                rejected_sources = preprocess_multimodal(copy.deepcopy([e["rejected"] for e in sources]),
                                                         self.data_args)
                chosen_data_dict = preprocess(chosen_sources, self.tokenizer, has_image=True)
                rejected_data_dict = preprocess(rejected_sources, self.tokenizer, has_image=True)
            else:
                chosen_sources = preprocess_multimodal(copy.deepcopy([e["chosen"] for e in sources]),
                                                       self.data_args)
                rejected_sources = preprocess_multimodal(copy.deepcopy([e["rejected"] for e in sources]),
                                                         self.data_args)
                chosen_data_dict = preprocess(chosen_sources, self.tokenizer, has_image=False)
                rejected_data_dict = preprocess(rejected_sources, self.tokenizer, has_image=False)

            if isinstance(i, int):
                data_dict = dict(
                    chosen_input_ids=chosen_data_dict["input_ids"][0],
                    chosen_labels=chosen_data_dict["labels"][0],
                    rejected_input_ids=rejected_data_dict["input_ids"][0],
                    rejected_labels=rejected_data_dict["labels"][0],
                )

            # image exists in the data
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # the image does not exist in the data, but the model is multimodal
                if hasattr(self.data_args.image_processor, 'crop_size'):
                    crop_size = self.data_args.image_processor.crop_size
                    data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
                else:
                    size = self.data_args.image_processor.size
                    data_dict['image'] = [torch.zeros(3, size['height'], size['width'])]
            return data_dict
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__() - 1))


@dataclass
class DataCollatorForDPODataset(object):
    """Collate examples for dpo training."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels = tuple([instance[key] for instance in instances]
                                                for key in ("chosen_input_ids", "chosen_labels"))
        # print('before Collator', input_ids)
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        chosen_labels = torch.nn.utils.rnn.pad_sequence(
            chosen_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        rejected_input_ids, rejected_labels = tuple([instance[key] for instance in instances]
                                                    for key in ("rejected_input_ids", "rejected_labels"))
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        rejected_labels = torch.nn.utils.rnn.pad_sequence(
            rejected_labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        # FIXME: chosen + rejected <= model_max_length
        batch = dict(
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
            rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
        )

        '''
            will be converted to a sequence of list, if batch size=6:
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(8, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
        '''
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # adapt to multi-video or multi-image or multi-image & video
            new_images = []
            for image in images:
                if type(image) is list:
                    for i in image:
                        new_images.append(i)
                else:
                    new_images.append(image)
            images = new_images

            # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
            batch['images'] = images
        else:
            raise ValueError(f'pretrain, {instances}')

        return batch


def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                         data_args) -> Dict:
    """Make dataset and collator for dpo training."""
    train_dataset = LazyDPODataset(tokenizer=tokenizer,
                                   data_path=data_args.data_path,
                                   data_args=data_args)
    data_collator = DataCollatorForDPODataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
