#!/usr/bin/env bash
blender --background --python clevr_mv.py -- --use_gpu 1 --height 512 --width 512 \
--base_scene_blendfile data/base_scene_centered_light=4.blend --r_camera 10 \
--uniform_sample --num_view 100 --output_image_dir ./scene_2/train/ \
--transform_output_file ./scene_2/transforms_train.json --save_blendfiles 1 \
--output_blend_dir ./scene_2/train/scene.blend --output_instance_color ./scene_2/train/instance_color.json
