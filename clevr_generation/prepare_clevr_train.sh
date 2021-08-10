#!/usr/bin/env bash
blender --background --python clevr_mv.py -- --use_gpu 1 --height 800 --width 800 --rot_with_xyz --num_view 100 --output_image_dir ./scene_1/train/ --transform_output_file ./scene_1/transforms_train.json --save_blendfiles 1 --output_blend_dir ./scene_1/train/scene.blend --output_instance_list ./scene_1/train/instance_list.txt --output_instance_label ./scene_1/train/instance_label.txt
