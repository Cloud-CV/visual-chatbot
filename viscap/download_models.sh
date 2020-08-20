#!/usr/bin/env bash

cd viscap/

# download visdial model checkpoint and config
mkdir checkpoints/
wget -O checkpoints/lf_gen_mask_rcnn_x101_train_demo.pth https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/lf_gen_mask_rcnn_x101_train_demo.pth
wget -O data/visdial_1.0_word_counts_train.json https://s3.amazonaws.com/visual-dialog/data/v1.0/2019/visdial_1.0_word_counts_train.json

# download mask-rcnn, butd model checkpoints and configs
mkdir -p captioning/model_data
wget -O captioning/model_data/vocabulary_captioning_thresh5.txt https://dl.fbaipublicfiles.com/pythia/data/vocabulary_captioning_thresh5.txt
wget -O captioning/model_data/detectron_model.pth  https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth
wget -O captioning/model_data/butd.pth https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.pth
wget -O captioning/model_data/butd.yml https://dl.fbaipublicfiles.com/pythia/pretrained_models/coco_captions/butd.yml
wget -O captioning/model_data/detectron_model.yaml https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml
wget -O captioning/model_data/detectron_weights.tar.gz https://dl.fbaipublicfiles.com/pythia/data/detectron_weights.tar.gz
tar xf captioning/model_data/detectron_weights.tar.gz -C captioning/

cd ../
