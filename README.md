# head-center-crop
sudo apt install cmake

% install dlib https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf


conda create -n head python=3.9
conda install openmim
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install tqdm

pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmdet==2.28.2 mmpose==0.29.0
pip install anime-face-detector


conda install dlib
pip install face_recognition