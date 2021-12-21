# CALM: Collaborative Arabic Language Model
The CALM project is joint effort lead by [NCAI](https://sdaia.gov.sa/ncai/?Lang=en) in collaboration with [Yandex](https://yandex.com/) and [HuggingFace](https://huggingface.co/) to train an Arabic language model with volunteers from around the globe. The project is an adaptation of the framework proposed at the NeurIPS 2021 demonstration: [Training Transformers Together](https://huggingface.co/training-transformers-together). 


Once of the main obstacles facing many researchers in the Arabic NLP community is the lack of computing resources that are needed for training large models. Models with leading performane on Arabic NLP tasks, such as [AraBERT](https://github.com/aub-mind/arabert), [CamelBERT](https://github.com/CAMeL-Lab/CAMeLBERT), [AraELECTRA](https://huggingface.co/aubmindlab/araelectra-base-generator), and [QARiB](https://huggingface.co/qarib), took days to train on TPUs. In the spirit of democratization of AI and community enabling, a core value at NCAI, CALM aims to demonstrate the effectiveness of collaborative training and form a community of volunteers for ANLP researchers with basic level cloud GPUs who wish to train their own models collaboratively. 

CALM trains a single BERT model on a dataset that combines MSA, Oscar and Arabic Wikipedia, and dialectal data for the gulf region from existing open source datasets. Each volunteer GPU trains the model locally at its own pace on a portion of the dataset while another portion is being streamed in the background to reduces local memory consumption. Computing the gradients and aggregating them is performed in a distributed manner, based on the computing abilities of each participating volunteer. Details of the distributed training process are further described in the paper [Deep Learning in Open Collaborations](https://papers.nips.cc/paper/2021/hash/41a60377ba920919939d83326ebee5a1-Abstract.html).

## How to host an experiment?

You will need:

### Set up auxiliary peers
You will need 1-3 workers that track metrics, upload statistics, etc. These peers do not use GPU.


Setup env:
```
sudo apt install -y git unzip tmux
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh > Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p ~/anaconda3
source ~/anaconda3/bin/activate
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone https://github.com/NCAI-Research/CALM/
pip install https://github.com/learning-at-home/hivemind/archive/calm.zip
cd calm && pip install -q -r requirements.txt &> log

# re-install bitsandbytes for the actual CUDA version
pip uninstall -y bitsandbytes-cuda111
pip install -y bitsandbytes-cuda113==0.26.0
```


Run auxiliary worker:

1. Open a tmux (or screen) session that will stay up after you logout. (`tmux new` , [about tmux](https://tmuxcheatsheet.com/))
2. Generate peer ID ahead of time

```bash
curl -L https://www.dropbox.com/s/p1hi93ahy5295jf/p2p-keygen?dl=1 > p2p-keygen
chmod +x p2p-keygen
./p2p-keygen -f ./identity
```
This ensures that if you restart the peer during, it will have the same identity, which is useful if others use your worker as initial peer.

3. Set environment variables
```bash
export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
export PORT_THAT_I_OPENED=12345
# ^-- please ensure you can accept incoming tcp connections on this port

export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT_THAT_I_OPENED
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT_THAT_I_OPENED
export CUDA_VISIBLE_DEVICES=

# organizations
export WANDB_ENTITY=CALM
export HF_ORGANIZATION_NAME=CALM

# experiment name
export EXP_NAME=CALM
export WANDB_PROJECT=$EXP_NAME
export HF_MODEL_NAME=$EXP_NAME
export WANDB_API_KEY=TODO_get_your_wandb_key_here_https://wpython run_aux_peer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --wandb_project $WANDB_PROJECT --identity ./identity --store_checkpoints --upload_interval 43200 --repo_url $HF_ORGANIZATION_NAME/$HF_MODEL_NAME --authorizeandb.ai/authorize
export HF_USER_ACCESS_TOKEN=TODO_create_user_access_token_here_with_WRITE_permissions_https://huggingface.co/settings/token

```

```bash
export WANDB_START_METHOD=thread
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT_THAT_I_OPENED
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT_THAT_I_OPENED
export CUDA_VISIBLE_DEVICES=

ulimit -n 16384
# ^-- this part is important

python run_aux_peer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --wandb_project $WANDB_PROJECT --identity ./identity --store_checkpoints --upload_interval 43200 --repo_url $HF_ORGANIZATION_NAME/$HF_MODEL_NAME --authorize
# Optionally, add more peers to the training via `--initial_peers ONE_OR_MORE PEERS_HERE`
```

If everything went right, it will print its address as such:
![image](https://user-images.githubusercontent.com/3491902/146950956-0ea06e77-15b4-423f-aeaa-02eb6aec06db.png)


Please copy this address and use it as ``--initial_peers`` with GPU/TPU trainers and other auxiliary peers.
