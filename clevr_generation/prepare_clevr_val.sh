#!/usr/bin/env bash
blender --background --python clevr_mv.py -- --use_gpu 1 --height 512 --width 512 \
--saved_blendfile ./scene_2/train/scene.blend --r_camera 10.5 \
--uniform_sample --num_view 50 --output_image_dir ./scene_2/val/ \
--transform_output_file ./scene_2/transforms_val.json --render_from_savedfile \
--saved_instance_color ./scene_2/train/instance_color.json
