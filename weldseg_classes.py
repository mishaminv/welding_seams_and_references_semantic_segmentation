import torch
import torchvision
import numpy
import ndimage


class weldSegmentatorClass(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        model_h_size: int,
        crop_width: int,
        cross_width: int,
        mirror_pad: int,
        mean: float,
        std: float,
        splitter: torch.nn.Module = None,
        min_crop_width: int = None,
        max_crop_width: int = None,
        p_threshold: float = 0.5,
        split_by_edges: bool = False,
        max_batch_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model_h_size = model_h_size
        self.crop_width = crop_width

        self.min_crop_width = crop_width if min_crop_width is None else min_crop_width
        self.max_crop_width = crop_width if max_crop_width is None else max_crop_width

        self.cross_width = cross_width
        self.mirror_pad = mirror_pad

        self.mean = mean
        self.std = std

        if 0 <= p_threshold <= 1.0:
            self.p_threshold = p_threshold
        else:
            raise ValueError("p_threshold must be in [0,1]")

        if split_by_edges and splitter is None:
            raise ValueError("split_by_edges set to True, but splitter not defined")
        
        if splitter is not None:
            self.splitter = splitter
            self.splitter.eval()

        self.split_by_edges = split_by_edges
        self.max_batch_size = max_batch_size

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """
        image : [1, H, W]
        """
        NUM_CLASSES = 1
        IMAGE_CHANNELS = 1

        image = (image - self.mean) / self.std

        crop_width = self.crop_width
        cross_width = self.cross_width
        batch_size = self.max_batch_size
        mirror_pad = self.mirror_pad
        max_crop_width = self.max_crop_width
        min_crop_width = self.min_crop_width
        model_device = next(self.model.parameters()).device

        orig_image_size = (image.size(-2), image.size(-1))
        if self.model_h_size != image.size(-2):
            aspect_ratio = image.size(-1) / image.size(-2)
            image = torchvision.transforms.functional.resize(
                image,
                size=(self.model_h_size, int(0.5 + self.model_h_size * aspect_ratio)),
            )

        if mirror_pad > 0:
            image = torch.nn.ReflectionPad2d((mirror_pad, mirror_pad, 0, 0))(image)

        # if image fits in single crop
        if image.size(-1) <= max_crop_width:
            if image.size(-1) < min_crop_width:
                pad_needed = min_crop_width - image.size(-1)
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad
                image = torch.nn.ReflectionPad2d((left_pad, right_pad, 0, 0))(image)
            else:
                left_pad = 0
                right_pad = 0

            image_device = image.device
            image = image.to(model_device)
            with torch.no_grad():
                proba = self.model(image.unsqueeze(0)).squeeze(0)
            proba = proba[
                ..., left_pad + mirror_pad : proba.size(-1) - (right_pad + mirror_pad)
            ].to(image_device)

            if proba.size(-2) != orig_image_size[0]:
                proba = torchvision.transforms.functional.resize(
                    proba, size=orig_image_size
                )
            return proba
        else:
            # more than 1 crops
            remainder = (image.size(-1) - crop_width) % (crop_width - cross_width)

            pad_needed = (
                ((crop_width - cross_width) - remainder) if remainder > 0 else 0
            )
            right_pad = pad_needed // 2
            left_pad = pad_needed - right_pad

            if left_pad > 0:
                image = torch.nn.ReflectionPad2d((left_pad, right_pad, 0, 0))(image)

            image_width = image.size(-1)
            image_height = image.size(-2)

            crops_num = ((image_width - crop_width) // (crop_width - cross_width)) + 1

            proba = torch.empty(
                size=(crops_num, NUM_CLASSES, image_height, crop_width),
                dtype=torch.float,
                device=image.device,
            )

            crops_processed = 0
            while crops_processed < crops_num:
                batch = torch.empty(
                    size=(
                        min(batch_size, crops_num - crops_processed),
                        IMAGE_CHANNELS,
                        image_height,
                        crop_width,
                    ),
                    dtype=torch.float,
                    device=image.device,
                )
                i = 0
                for i in range(batch.size(0)):
                    batch[i, ...] = image[
                        ...,
                        (crop_width - cross_width)
                        * crops_processed : (crop_width - cross_width)
                        * crops_processed
                        + crop_width,
                    ]
                    crops_processed += 1
                batch = batch.to(next(self.model.parameters()).device)
                with torch.no_grad():
                    proba[
                        crops_processed - batch.size(0) : crops_processed, ...
                    ] = self.model(batch).to(
                        image.device
                    )  # [N, 2, H, W]
                del batch

            left_smother = torch.arange(
                start=1, end=cross_width + 1, dtype=torch.float, device=image.device
            ) / (cross_width + 1)
            right_smother = (
                reversed(left_smother).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            left_smother = left_smother.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # smothing intersections
            proba[1:, :, :, :cross_width] *= left_smother.expand(
                proba.size(0) - 1, NUM_CLASSES, image_height, cross_width
            )
            proba[:-1, :, :, proba.size(-1) - cross_width :] *= right_smother.expand(
                proba.size(0) - 1, NUM_CLASSES, image_height, cross_width
            )
            del left_smother, right_smother

            # combining probabilities
            proba[1:, :, :, :cross_width] += proba[:-1, :, :, -cross_width:]

            # connecting part
            proba_combined = torch.empty(
                size=(
                    NUM_CLASSES,
                    image_height,
                    image_width - (left_pad + right_pad + mirror_pad * 2),
                ),
                dtype=torch.float,
                device=image.device,
            )
            # remove left_pad from the firts crop
            proba_combined[
                ..., : crop_width - cross_width - (left_pad + mirror_pad)
            ] = proba[0, :, :, (left_pad + mirror_pad) : (crop_width - cross_width)]
            for i in range(1, crops_num - 1):
                proba_combined[
                    ...,
                    crop_width
                    - cross_width
                    - (left_pad + mirror_pad)
                    + (i - 1) * (crop_width - cross_width) : crop_width
                    - cross_width
                    - (left_pad + mirror_pad)
                    + i * (crop_width - cross_width),
                ] = proba[i, :, :, : proba.size(-1) - cross_width]
            # remove right_pad from the last crop
            i = crops_num - 1
            proba_combined[
                ...,
                crop_width
                - cross_width
                - (left_pad + mirror_pad)
                + (i - 1) * (crop_width - cross_width) :,
            ] = proba[i, :, :, : proba.size(-1) - (right_pad + mirror_pad)]

            del proba

            if proba_combined.size(-2) != orig_image_size[0]:
                proba_combined = torchvision.transforms.functional.resize(
                    proba_combined, size=orig_image_size
                )
            return proba_combined

    def make_mask(self, probas: torch.Tensor, treshhold: float = None) -> torch.Tensor:
        """
        probas : [1, H, W]

        returns [H, W]
        """
        if treshhold is None:
            treshhold = self.p_threshold
        prediction = probas > treshhold
        del probas

        device = prediction.device
        prediction = prediction.to("cpu").numpy()
        prediction = prediction.squeeze(0)

        labels, num = ndimage.label(prediction)  # 4-связность по умолчанию

        if num == 0:
            return torch.tensor(prediction).to(device).long()
        else:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0  # фон не считаем

            largest_label = sizes.argmax()
            return torch.tensor(labels == largest_label).to(device).long()

    def forward(self, image: torch.tensor) -> torch.tensor:
        """
        image : [1, H, W]
        """
        image_device = image.device

        if self.split_by_edges:
            splitter_device = next(self.splitter.parameters()).device
            with torch.no_grad():
                parts = self.splitter(image.to(splitter_device))
            del image
        else:
            parts = [image]

        for i in range(len(parts)):
            parts[i] = self.predict(parts[i].to(image_device))

        if len(parts) > 1:
            proba = torch.cat(parts, dim=-1)
        else:
            proba = parts[0]

        return self.make_mask(proba)

class weldSegmentator2StageClass(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        model_h_size: int,
        crop_width: int,
        cross_width: int,
        mirror_pad: int,
        mean: int,
        std: int,
        weld_detector: torch.nn.Module,
        weld_detector_h_size: int,
        splitter: torch.nn.Module = None,
        weld_detector_acceptable_discrepancy: float = 0.05,
        weld_detector_min_wide_filter: float | int = 0.08,
        weld_detector_p_level: float = 0.5,
        min_crop_width: int = None,
        max_crop_width: int = None,
        weld_detector_min_crop_width: int = None,
        weld_detector_crop_width: int = 128,
        weld_detector_max_crop_width: int = None,
        weld_box_expand_coef: float = 1.61271,
        p_threshold: float = 0.5,
        split_by_edges: bool = False,
        max_batch_size: int = 16,
        weld_detector_max_batch_size: int = 16,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model_h_size = model_h_size
        self.crop_width = crop_width

        self.min_crop_width = crop_width if min_crop_width is None else min_crop_width
        self.max_crop_width = crop_width if max_crop_width is None else max_crop_width

        self.cross_width = cross_width
        self.mirror_pad = mirror_pad

        self.mean = mean
        self.std = std

        if 0 <= weld_detector_p_level <= 1.0:
            self.weld_detector_p_level = weld_detector_p_level
        else:
            raise ValueError("weld_detector_p_level must be in [0,1]")

        if 0 <= p_threshold <= 1.0:
            self.p_threshold = p_threshold
        else:
            raise ValueError("p_threshold must be in [0,1]")

        if split_by_edges and splitter is None:
            raise ValueError("split_by_edges set to True, but splitter not defined")

        if splitter is not None:
            self.splitter = splitter
            self.splitter.eval()
        self.split_by_edges = split_by_edges

        self.weld_box_expand_coef = weld_box_expand_coef
        self.weld_detector = weld_detector
        self.weld_detector.eval()
        self.weld_detector_h_size = weld_detector_h_size
        self.weld_detector_acceptable_discrepancy = weld_detector_acceptable_discrepancy
        self.weld_detector_crop_width = weld_detector_crop_width

        self.weld_detector_min_crop_width = (
            weld_detector_crop_width
            if weld_detector_min_crop_width is None
            else weld_detector_min_crop_width
        )
        self.weld_detector_max_crop_width = (
            weld_detector_crop_width
            if weld_detector_max_crop_width is None
            else weld_detector_max_crop_width
        )

        self.weld_detector_min_wide_filter = weld_detector_min_wide_filter

        self.max_batch_size = max_batch_size
        self.weld_detector_max_batch_size = weld_detector_max_batch_size

    def weld_selector(self, pred: torch.tensor):
        """
        pred : [im_H]
        """

        p_level = self.weld_detector_p_level
        min_wide_filter = self.weld_detector_min_wide_filter

        p_max = None
        jj = 0
        N = pred.numel()
        if type(min_wide_filter) == float:
            min_wide_filter = int(N * min_wide_filter)

        while jj < N:
            while (jj < N) and (pred[jj] <= p_level):
                jj += 1

            if jj < N:
                weld_start = jj
                p = pred[jj].item()
                l = 1
                jj += 1
                while (jj < N) and (pred[jj] > p_level):
                    p += pred[jj]
                    l += 1
                    jj += 1

                if l > min_wide_filter and (p_max is None or (p / l) > p_max):
                    p_max = p / l
                    p_max_weld_start = weld_start
                    p_max_weld_end = jj - 1

        if p_max is not None:
            return p_max_weld_start, p_max_weld_end + 1
        else:
            return 0, 0

    def get_weld_boxes(self, image : torch.Tensor) -> list[dict]:
        IMAGE_CHANNELS = 1
        crop_width = self.weld_detector_crop_width
        max_crop_width = self.weld_detector_max_crop_width
        min_crop_width = self.weld_detector_min_crop_width
        detector_h_size = self.weld_detector_h_size
        batch_size = self.weld_detector_max_batch_size
        weld_detector_device = next(self.weld_detector.parameters()).device
        cross_width = int(
            self.cross_width * (self.weld_detector_h_size / self.model_h_size)
        )

        orig_image = image
        if detector_h_size != image.size(-2):
            decompres_ratio = image.size(-2) / detector_h_size
            aspect_ratio = image.size(-1) / image.size(-2)
            orig_image_w = image.size(-1)
            image = torchvision.transforms.functional.resize(
                image, size=(detector_h_size, int(0.5 + detector_h_size * aspect_ratio))
            )

        # if image fits in single crop
        if image.size(-1) <= max_crop_width:
            if image.size(-1) < min_crop_width:
                pad_needed = min_crop_width - image.size(-1)
                left_pad = pad_needed // 2
                right_pad = pad_needed - left_pad
                image = torch.nn.ReflectionPad2d((left_pad, right_pad, 0, 0))(image)

            image_device = image.device
            image = image.to(weld_detector_device)
            image = (image - self.mean) / self.std
            probas = self.weld_detector(image.unsqueeze(0)).squeeze(0).to(image_device)
            lower_bound, upper_bound = self.weld_selector(probas)

            del image, probas

            lower_bound = int(0.5 + lower_bound * decompres_ratio)
            upper_bound = int(0.5 + upper_bound * decompres_ratio)

            box_h_increase = int(
                0.5 + (upper_bound - lower_bound) * (self.weld_box_expand_coef - 1.0)
            )
            to_lower = box_h_increase // 2
            to_upper = box_h_increase - to_lower

            expanded_lower_bound = lower_bound - to_lower
            if expanded_lower_bound < 0:
                expanded_lower_bound = 0

            expanded_upper_bound = upper_bound + to_upper
            if expanded_upper_bound > orig_image.size(-2):
                expanded_upper_bound = orig_image.size(-2)

            image = torchvision.transforms.functional.resize(
                orig_image[
                    ..., expanded_lower_bound:expanded_upper_bound, 0:orig_image_w
                ],
                size=(
                    self.model_h_size,
                    int(0.5 + orig_image_w * self.model_h_size / orig_image.size(-2)),
                ),
            )

            result = {
                "left": 0,
                "right": orig_image_w,
                "lower_bound": expanded_lower_bound,
                "upper_bound": expanded_upper_bound,
                "image": image,
            }

            return [result]
        # more then 1 crop
        else:
            remainder = (image.size(-1) - crop_width) % (crop_width - cross_width)
            pad_needed = (
                ((crop_width - cross_width) - remainder) if remainder > 0 else 0
            )
            if pad_needed > 0:
                image = torch.nn.ReflectionPad2d((0, pad_needed, 0, 0))(image)

            image_width = image.size(-1)
            image_height = image.size(-2)

            crops_num = ((image_width - crop_width) // (crop_width - cross_width)) + 1

            weld_boxes = []

            crops_processed = 0
            while crops_processed < crops_num:
                batch = torch.empty(
                    size=(
                        min(batch_size, crops_num - crops_processed),
                        IMAGE_CHANNELS,
                        image_height,
                        crop_width,
                    ),
                    dtype=torch.float,
                    device=image.device,
                )
                part_id = 0
                for part_id in range(batch.size(0)):
                    batch[part_id, ...] = image[
                        ...,
                        (crop_width - cross_width)
                        * crops_processed : (crop_width - cross_width)
                        * crops_processed
                        + crop_width,
                    ]
                    crops_processed += 1
                batch = batch.to(weld_detector_device)
                batch = (batch - self.mean) / self.std
                probas_batch = self.weld_detector(batch).to(image.device)
                for part_id in range(batch.size(0)):
                    lower_bound, upper_bound = self.weld_selector(probas_batch[part_id])

                    weld_boxes.append(
                        {
                            "left": (crop_width - cross_width)
                            * (crops_processed - batch.size(0) + part_id),
                            "right": crop_width
                            + (crop_width - cross_width)
                            * (crops_processed - batch.size(0) + part_id),
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                        }
                    )

                del batch, probas_batch

            min_weld_h = weld_boxes[0]["upper_bound"] - weld_boxes[0]["lower_bound"]

            # concat boxes if it posible
            part_id = 1
            while part_id < len(weld_boxes):
                new_lower = min(
                    weld_boxes[part_id - 1]["lower_bound"],
                    weld_boxes[part_id]["lower_bound"],
                )
                new_upper = max(
                    weld_boxes[part_id - 1]["upper_bound"],
                    weld_boxes[part_id]["upper_bound"],
                )

                min_weld_h = min(
                    min_weld_h,
                    weld_boxes[part_id]["upper_bound"]
                    - weld_boxes[part_id]["lower_bound"],
                )

                if (
                    1.0 - (min_weld_h / (new_upper - new_lower))
                    < self.weld_detector_acceptable_discrepancy
                ):
                    weld_boxes[part_id - 1]["right"] = weld_boxes[part_id]["right"]
                    weld_boxes[part_id - 1]["upper_bound"] = new_upper
                    weld_boxes[part_id - 1]["lower_bound"] = new_lower
                    weld_boxes.pop(part_id)
                else:
                    min_weld_h = (
                        weld_boxes[part_id]["upper_bound"]
                        - weld_boxes[part_id]["lower_bound"]
                    )
                    part_id += 1

            for box in weld_boxes:
                box["left"] = int(box["left"] * decompres_ratio + 0.5)
                box["right"] = int(box["right"] * decompres_ratio + 0.5)
                box["lower_bound"] = int(box["lower_bound"] * decompres_ratio + 0.5)
                box["upper_bound"] = int(box["upper_bound"] * decompres_ratio + 0.5)

            weld_boxes[len(weld_boxes) - 1]["right"] = orig_image_w

            for box in weld_boxes:

                box_h_increase = int(
                    0.5
                    + (box["upper_bound"] - box["lower_bound"])
                    * (self.weld_box_expand_coef - 1.0)
                )
                to_lower = box_h_increase // 2
                to_upper = box_h_increase - to_lower

                expanded_lower_bound = box["lower_bound"] - to_lower
                if expanded_lower_bound < 0:
                    expanded_lower_bound = 0

                box["lower_bound"] = expanded_lower_bound

                expanded_upper_bound = box["upper_bound"] + to_upper
                if expanded_upper_bound > orig_image.size(-2):
                    expanded_upper_bound = orig_image.size(-2)
                box["upper_bound"] = expanded_upper_bound

                image = torchvision.transforms.functional.resize(
                    orig_image[
                        ...,
                        box["lower_bound"] : box["upper_bound"],
                        box["left"] : box["right"],
                    ],
                    size=(
                        self.model_h_size,
                        int(
                            0.5
                            + (box["right"] - box["left"])
                            * self.model_h_size
                            / orig_image.size(-2)
                        ),
                    ),
                )

                box["image"] = image

            return weld_boxes

    def predict_engine(self, image : torch.Tensor, mirror_pad: tuple[int, int]=(0, 0), smooth_edges : tuple[bool, bool]=(True, True))-> torch.Tensor:
        """
        image : [1, H, W]
        """
        NUM_CLASSES = 1
        IMAGE_CHANNELS = 1

        crop_width = self.crop_width
        max_crop_width = self.max_crop_width
        min_crop_width = self.min_crop_width
        cross_width = self.cross_width
        batch_size = self.max_batch_size
        model_device = next(self.model.parameters()).device

        orig_image_size = (image.size(-2), image.size(-1))
        if self.model_h_size != image.size(-2):
            aspect_ratio = image.size(-1) / image.size(-2)
            image = torchvision.transforms.functional.resize(
                image,
                size=(self.model_h_size, int(0.5 + self.model_h_size * aspect_ratio)),
            )

        mirror_pad_left = mirror_pad[0]
        mirror_pad_right = mirror_pad[1]
        del mirror_pad

        image = torch.nn.ReflectionPad2d((mirror_pad_left, mirror_pad_right, 0, 0))(
            image
        )

        # if image fits in single crop
        if image.size(-1) <= max_crop_width:
            if image.size(-1) < min_crop_width:
                pad_needed = min_crop_width - image.size(-1)
                image = torch.nn.ReflectionPad2d((0, pad_needed, 0, 0))(image)
            else:
                pad_needed = 0

            image_device = image.device
            image = image.to(model_device)
            image = (image - self.mean) / self.std

            proba = self.model(image.unsqueeze(0)).squeeze(0)

            proba = proba[
                ..., mirror_pad_left : proba.size(-1) - (mirror_pad_right + pad_needed)
            ].to(image_device)
            if proba.size(-2) != orig_image_size[0]:
                proba = torchvision.transforms.functional.resize(
                    proba, size=orig_image_size
                )
            return proba

        # more than 1 crops
        else:
            remainder = (image.size(-1) - crop_width) % (crop_width - cross_width)

            pad_needed = (
                ((crop_width - cross_width) - remainder) if remainder > 0 else 0
            )

            if pad_needed > 0:
                image = torch.nn.ReflectionPad2d((0, pad_needed, 0, 0))(image)

            image_width = image.size(-1)
            image_height = image.size(-2)

            crops_num = ((image_width - crop_width) // (crop_width - cross_width)) + 1

            proba = torch.empty(
                size=(crops_num, NUM_CLASSES, image_height, crop_width),
                dtype=torch.float,
                device=image.device,
            )

            crops_processed = 0
            while crops_processed < crops_num:
                batch = torch.empty(
                    size=(
                        min(batch_size, crops_num - crops_processed),
                        IMAGE_CHANNELS,
                        image_height,
                        crop_width,
                    ),
                    dtype=torch.float,
                    device=image.device,
                )
                part_id = 0
                for part_id in range(batch.size(0)):
                    batch[part_id, ...] = image[
                        ...,
                        (crop_width - cross_width)
                        * crops_processed : (crop_width - cross_width)
                        * crops_processed
                        + crop_width,
                    ]
                    crops_processed += 1
                batch = batch.to(model_device)
                batch = (batch - self.mean) / self.std
                proba[
                    crops_processed - batch.size(0) : crops_processed, ...
                ] = self.model(batch).to(
                    image.device
                )  # [N, 2, H, W]
                del batch

            left_smother = torch.arange(
                start=1, end=cross_width + 1, dtype=torch.float, device=image.device
            ) / (cross_width + 1)
            right_smother = (
                reversed(left_smother).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            )
            left_smother = left_smother.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # smothing intersections
            proba[1:, :, :, :cross_width] *= left_smother.expand(
                proba.size(0) - 1, NUM_CLASSES, image_height, cross_width
            )
            proba[:-1, :, :, -cross_width:] *= right_smother.expand(
                proba.size(0) - 1, NUM_CLASSES, image_height, cross_width
            )

            del left_smother, right_smother

            # combining probabilities
            proba[1:, :, :, :cross_width] += proba[:-1, :, :, -cross_width:]

            # connecting parts
            proba_combined = torch.empty(
                size=(
                    NUM_CLASSES,
                    image_height,
                    image_width - (mirror_pad_left + mirror_pad_right + pad_needed),
                ),
                dtype=torch.float,
                device=image.device,
            )

            # remove left_pad from the firts crop
            proba_combined[..., : crop_width - cross_width - mirror_pad_left] = proba[
                0, :, :, mirror_pad_left : (crop_width - cross_width)
            ]
            for part_id in range(1, crops_num - 1):
                proba_combined[
                    ...,
                    crop_width
                    - cross_width
                    - mirror_pad_left
                    + (part_id - 1) * (crop_width - cross_width) : crop_width
                    - cross_width
                    - mirror_pad_left
                    + part_id * (crop_width - cross_width),
                ] = proba[part_id, :, :, :-cross_width]
            # remove right_pad from the last crop
            part_id = crops_num - 1
            proba_combined[
                ...,
                crop_width
                - cross_width
                - mirror_pad_left
                + (part_id - 1) * (crop_width - cross_width) :,
            ] = proba[part_id, :, :, : proba.size(-1) - (mirror_pad_right + pad_needed)]

            del proba

            # smothing edges
            left_smother = torch.arange(
                start=1, end=cross_width + 1, dtype=torch.float, device=image.device
            ) / (cross_width + 1)
            right_smother = reversed(left_smother).unsqueeze(0).unsqueeze(0)
            left_smother = left_smother.unsqueeze(0).unsqueeze(0)

            if smooth_edges[0]:
                proba_combined[..., :cross_width] *= left_smother
            if smooth_edges[1]:
                proba_combined[..., -cross_width:] *= right_smother

            del left_smother, right_smother

            if proba_combined.size(-2) != orig_image_size[0]:
                proba_combined = torchvision.transforms.functional.resize(
                    proba_combined, size=orig_image_size
                )
            return proba_combined

    def make_mask(self, probas, treshhold=None):
        """
        probas : [1, H, W]

        returns [H, W]
        """
        if treshhold is None:
            treshhold = self.p_threshold
        prediction = probas > treshhold
        del probas

        device = prediction.device
        prediction = prediction.to("cpu").numpy()
        prediction = prediction.squeeze(0)

        labels, num = ndimage.label(prediction)  # 4-связность по умолчанию

        if num == 0:
            result = prediction
        else:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0  # фон не считаем

            largest_label = sizes.argmax()
            result = labels == largest_label

        # result = ndimage.binary_fill_holes(result)

        return torch.tensor(result).to(device).long()


    def forward(self, image: torch.tensor) -> torch.tensor:
        with torch.no_grad():
            image_device = image.device
            if self.split_by_edges:
                splitter_device = next(self.splitter.parameters()).device
                parts = self.splitter(image.to(splitter_device))
                del image
            else:
                parts = [image]

            for part_id in range(len(parts)):
                part_size = parts[part_id].size()
                weld_boxes = self.get_weld_boxes(parts[part_id].to(image_device))
                parts[part_id] = None

                if len(weld_boxes) > 1:
                    box_id = 0
                    weld_boxes[box_id]["proba"] = self.predict_engine(
                        weld_boxes[box_id]["image"],
                        mirror_pad=(self.mirror_pad, 0),
                        smooth_edges=(False, True),
                    )
                    weld_boxes[box_id].pop("image")
                    for box_id in range(1, len(weld_boxes) - 1):
                        weld_boxes[box_id]["proba"] = self.predict_engine(
                            weld_boxes[box_id]["image"], smooth_edges=(True, True)
                        )
                        weld_boxes[box_id].pop("image")

                    box_id = len(weld_boxes) - 1
                    weld_boxes[box_id]["proba"] = self.predict_engine(
                        weld_boxes[box_id]["image"],
                        mirror_pad=(0, self.mirror_pad),
                        smooth_edges=(True, False),
                    )
                    weld_boxes[box_id].pop("image")
                else:
                    box_id = 0
                    weld_boxes[box_id]["proba"] = self.predict_engine(
                        weld_boxes[box_id]["image"],
                        mirror_pad=(self.mirror_pad, self.mirror_pad),
                        smooth_edges=(False, False),
                    )
                    weld_boxes[box_id].pop("image")

                proba_part = torch.zeros(
                    size=part_size, dtype=torch.float, device=image_device
                )
                for box in weld_boxes:
                    proba_part[
                        ...,
                        box["lower_bound"] : box["upper_bound"],
                        box["left"] : box["right"],
                    ] += torchvision.transforms.functional.resize(
                        box["proba"],
                        size=(
                            box["upper_bound"] - box["lower_bound"],
                            box["right"] - box["left"],
                        ),
                    )
                    box.pop("proba")

                parts[part_id] = proba_part

            proba = torch.cat(parts, dim=-1)

            prediction = self.make_mask(proba)
            return prediction


class edge_detector(torch.nn.Module):
    def __init__(
        self,
        min_w_to_h: float = 0.7,
    ):
        super().__init__()
        self.im_h = 128
        lines_h = 3
        self.min_w_to_h = min_w_to_h
        self.line_threshold = 0.003
        self.edge_threshold = 0.3
        self.line_detector_conv_1 = torch.nn.Conv2d(
            1, 1, kernel_size=(lines_h, 6), padding=(lines_h // 2, 2), bias=False
        )
        w = (
            torch.tensor(
                [
                    [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                ]
            )
            .unsqueeze(1)
            .unsqueeze(1)
            .repeat(1, 1, 3, 1)
        )
        self.line_detector_conv_1.weight = torch.nn.Parameter(w)

        self.line_detector_conv_2 = torch.nn.Conv2d(3, 1, kernel_size=1, bias=False)

        w = (
            torch.tensor([0.3333, -0.3333, -0.3333])
            .unsqueeze(1)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        self.line_detector_conv_2.weight = torch.nn.Parameter(w)

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        """
        x : [1, H, W]
        """
        with torch.no_grad():
            x = image.unsqueeze(0)
            x = torch.nn.functional.interpolate(x, size=(self.im_h, x.size(-1)))
            x = self.line_detector_conv_1(x)
            x = x.abs()
            x = self.line_detector_conv_2(x)
            x = x > self.line_threshold

            x = x.float().mean(dim=-2)
            # [1, 1, W - 1] -> [W - 1]
            x = x.squeeze()
            edges = x > self.edge_threshold
            edges[0] = False
            edges[-1] = False

            edges = torch.arange(start=1, end=edges.numel() + 1, device=edges.device)[
                edges
            ].tolist()

            # remove repeating detections
            edge_clusters_to_resolve = []
            i = 1
            while i < len(edges):
                while i < len(edges) and (edges[i] - edges[i - 1]) > 2:
                    i += 1
                if i < len(edges):
                    edge_clusters_to_resolve.append([])
                    edge_clusters_to_resolve[-1].append(i - 1)
                    while i < len(edges) and (edges[i] - edges[i - 1]) <= 2:
                        edge_clusters_to_resolve[-1].append(i)
                        i += 1

            for edge_cluster in reversed(edge_clusters_to_resolve):
                edge_intensities = [x[edges[idx] - 1].item() for idx in edge_cluster]
                keep_edge_idx = edge_cluster[
                    edge_intensities.index(max(edge_intensities))
                ]

                # removing others
                for removing_idx in reversed(edge_cluster):
                    if removing_idx != keep_edge_idx:
                        edges.pop(removing_idx)

            # remove too small crops
            while len(edges) > 0 and edges[0] < int(self.min_w_to_h * image.size(-2)):
                edges.pop(0)
            while len(edges) > 0 and (image.size(-1) - edges[-1]) < int(
                self.min_w_to_h * image.size(-2)
            ):
                edges.pop(-1)

            i = 1
            while i < len(edges):
                if (edges[i] - edges[i - 1]) < int(self.min_w_to_h * image.size(-2)):
                    # x stores edge intensivities
                    if x[edges[i] - 1] > x[edges[i - 1] - 1]:
                        edges.pop(i - 1)
                    else:
                        edges.pop(i)
                else:
                    i += 1

            edges = [0] + edges + [image.size(-1)]
            result = []

            for i in range(1, len(edges)):
                result.append(image[..., edges[i - 1] : edges[i]].clone())

            return result


class box_predictor(torch.nn.Module):
    def __init__(self, pool="max"):
        super().__init__()

        if pool == "max":
            pool_layer = torch.nn.AdaptiveMaxPool2d(output_size=(1, 1))
        elif pool == "avg":
            pool_layer = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.head = torch.nn.Sequential(pool_layer, torch.nn.Sigmoid())

        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(),
            torch.nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=2, stride=1, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.body(x)
        x = self.head(x)
        return x.squeeze(3).squeeze(2)
