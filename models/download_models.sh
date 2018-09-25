# neuraltalk2
cd neuraltalk2
wget http://cs.stanford.edu/people/karpathy/neuraltalk2/checkpoint_v1.zip
unzip checkpoint_v1.zip
cd ..

cd models
wget https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/ded9363bd93ec0c770134f4e387d8aaaaa2407ce/VGG_ILSVRC_16_layers_deploy.prototxt
wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
wget https://s3.amazonaws.com/visual-dialog/models/v0.9/hre-ques-im-hist-gen-vgg16-14.t7
cd ..

mkdir data && cd data
wget https://s3.amazonaws.com/cvmlp/visdial-pytorch/data/chat_processed_params_0.9.json
cd ..
