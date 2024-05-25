pwd
cd ../
mkdir download
cd ./download/
wget https://github.com/vllm-project/vllm-nccl/releases/download/v0.1.0/cu12-libnccl.so.2.18.1
ls
cp cu12-libnccl.so.2.18.1 /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1
ls /root/.config/vllm/nccl/cu12/libnccl.so.2.18.1
cd ~
pip install -e .
stat /usr/local/cuda
/etc/alternatives/cuda/bin/nvcc --version
CUDACXX=/etc/alternatives/cuda/bin/nvcc pip install -e .
pip install spacy=0.10
pip install spacy==0.10
pip install typer==
pip install typer==0.10.0
pip install typer==0.9.4
pip install fastapi-cli==
pip install fastapi-cli==0.0.1
pip install fastapi-cli==0.0.2
torch --version
pip list
ls
cd /download/
cd -
python examples/offline_inference.py 
pwd
exit
ls
cd ../
ls
su -
exit
cd /root
ls
cd vllm
cd ..
pip install -e .
cd examples/
ls
python ./offline_inference.py 
cd ..
pip install -e .
nvcc --version
pip install -e .
python -m pip install --upgrade pip
ls
cd ../
ls
su -
