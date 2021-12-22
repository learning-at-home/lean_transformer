# CALM: Collaborative Arabic Language Model
The CALM project is joint effort lead by [NCAI](https://sdaia.gov.sa/ncai/?Lang=en) in collaboration with [Yandex](https://yandex.com/) and [HuggingFace](https://huggingface.co/) to train an Arabic language model with volunteers from around the globe. The project is an adaptation of the framework proposed at the NeurIPS 2021 demonstration: [Training Transformers Together](https://huggingface.co/training-transformers-together). 


Once of the main obstacles facing many researchers in the Arabic NLP community is the lack of computing resources that are needed for training large models. Models with leading performane on Arabic NLP tasks, such as [AraBERT](https://github.com/aub-mind/arabert), [CamelBERT](https://github.com/CAMeL-Lab/CAMeLBERT), [AraELECTRA](https://huggingface.co/aubmindlab/araelectra-base-generator), and [QARiB](https://huggingface.co/qarib), took days to train on TPUs. In the spirit of democratization of AI and community enabling, a core value at NCAI, CALM aims to demonstrate the effectiveness of collaborative training and form a community of volunteers for ANLP researchers with basic level cloud GPUs who wish to train their own models collaboratively. 

CALM trains a single BERT model on a dataset that combines MSA, Oscar and Arabic Wikipedia, and dialectal data for the gulf region from existing open source datasets. Each volunteer GPU trains the model locally at its own pace on a portion of the dataset while another portion is being streamed in the background to reduces local memory consumption. Computing the gradients and aggregating them is performed in a distributed manner, based on the computing abilities of each participating volunteer. Details of the distributed training process are further described in the paper [Deep Learning in Open Collaborations](https://papers.nips.cc/paper/2021/hash/41a60377ba920919939d83326ebee5a1-Abstract.html).

## How to host your own experiment?

Here you can find the best practices that we learned from running the CALM experiment. These may help you set up your own collaborative experiment.

If your training run is not confidential, feel free to ask for help on the [hivemind discord channel](https://discord.gg/uGugx9zYvN)

<details>
  <summary><b> 1. Choose and verify your training configuration</b></summary>  
  
  Depending on you use case, you may want to change
   - Dataset and preprocessing ([`data.py`](https://github.com/NCAI-Research/CALM/blob/main/tasks/mlm/data.py), [`data_cleaning.py`](https://github.com/NCAI-Research/CALM/blob/main/tasks/mlm/data_cleaning.py), [`whole_word_mask.py`](https://github.com/NCAI-Research/CALM/blob/main/tasks/mlm/whole_word_mask.py);
   - Tokenizer (see [`arguments.py`](https://github.com/NCAI-Research/CALM/blob/main/arguments.py#L110-L112))
   - Model config ([`model.json`](https://github.com/NCAI-Research/CALM/blob/main/tasks/mlm/model.json)
  
  
  When transitioning to a new language or new dataset, it is important to check that the tokenizer/collator works as intended **before** you begin training.
  The best way to do that is to manually look at training minibatches:
  ```python
  from tasks.mlm.data import make_training_dataset
  from tasks.mlm.whole_word_mask import DataCollatorForWholeWordMask
  
  tokenizer = create_tokenizer_here(...)
  dataset = make_training_dataset(tokenizer, max_sequence_length=...)  # see arguments.py
  collator = DataCollatorForWholeWordMask(tokenizer, pad_to_multiple_of=...)  # see arguments.py
  data_loader = torch.utils.data.DataLoader(dataset, collate_fn=collator, batch_size=4)

  # generate a few batches
  rows = []
  with tqdm(enumerate(data_loader)) as progress:
      for i, row in progress:
          rows.append(row)
          if i > 10:
              break
  
  # look into the training data
  row_ix, sample_ix = 0, 1
  sources = [tokenizer.decode([i]) for i in rows[row_ix]['input_ids'][sample_ix].data.numpy()]
  print("MASK RATE:", (rows[row_ix]['input_ids'][sample_ix] == 4).data.numpy().sum() / (rows[row_ix]['input_ids'][sample_ix] != 0).data.numpy().sum())

  for i in range(len(sources)):
      if sources[i] == '[MASK]':
          pass#sources[i] = '[[' + tokenizer.decode(rows[row_ix]['labels'][sample_ix][i].item()) + ']]'

  print(' '.join(sources))
  ```
  
  If you make many changes, it also helps to train a very model using your own device to check if everything works as intended. A good initial configuration is 6 layers, 512 hidden, 2048 intermediate).
  
  If you're training with volunteers, the most convenient way is to set up a Hugging Face organization. For instructions on that, see "make your own" section of https://training-transformers-together.github.io . We use WANDB for tracking logs and training progress: we've set up a [WandB team](https://docs.wandb.ai/ref/app/features/teams) named [CALM](https://wandb.ai/calm) for this experiment. Alternatively, you can use hivemind standalone (and even without internet access) by setting --authorize False and WANDB_DISABLED=true -- or manually removing the corresponding options from the code.
 
</details>

<details>
  <summary> <b>2. Setting up auxiliary peers</b> </summary>

Auxiliary peers are low-end servers without GPU that will keep track of the latest model checkpoint and report metrics and assist in communication.
You will need 1-3 workers that track metrics, upload statistics, etc. These peers do not use GPU.
If you have many participants are behind firewall (in --client_mode), it helps to add more auxiliary servers, as they can serve as relays and help with all-reduce.
  
__Minimum requirements:__ 15+ GB RAM, at least 100Mbit/s download/upload speed, at least one port opened to incoming connections;

__Where to get:__ cloud providers that have cheap ingress/egress pricing. Good examples: [pebblehost](https://pebblehost.com/dedicated/) and [hetzner](https://console.hetzner.cloud/). AWS/GCP/Azure are more convenient, but they cost more due to egress pricing. Path of the true jedi: use your homelab or university server -- but that may require networking experience.



__Setup env:__

```
sudo apt install -y git tmux
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh > Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p ~/anaconda3
source ~/anaconda3/bin/activate
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone https://github.com/NCAI-Research/CALM/
pip install https://github.com/learning-at-home/hivemind/archive/calm.zip
cd CALM && pip install -q -r requirements.txt &> log

# re-install bitsandbytes for the actual CUDA version
pip uninstall -y bitsandbytes-cuda111
pip install -y bitsandbytes-cuda113==0.26.0

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```


__Run auxiliary worker:__

1. Open a tmux (or screen) session that will stay up after you logout. (`tmux new` , [about tmux](https://tmuxcheatsheet.com/))
2. Generate peer ID ahead of time

```bash
curl -L https://www.dropbox.com/s/p1hi93ahy5295jf/p2p-keygen?dl=1 > p2p-keygen
chmod +x p2p-keygen
./p2p-keygen -f ./identity


```
This ensures that if you restart the peer during training, it will have the same identity, which is useful if others use your worker as initial peer.
  
3. Measure internet bandwidth and set `$BANDWIDTH` variable
```bash

# You can measure bandwidth automatically:
curl -s https://gist.githubusercontent.com/justheuristic/5467799d8f2ad59b36fa75f642cc9b87/raw/c5a4b9b66987c2115e6c54a07d97e0104dfbcd97/speedtest.py | python -  --json > speedtest.json
export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`
echo "Internet Bandwidth (Mb/s) = $BANDWIDTH"
  
# If that doesn't work, you can simply `export BANDWIDTH=TODOyour_bandwidth_mbits_here` using the minimum of download and upload speed.
```
  

4. Run the auxiliary peer
```bash
export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
export PORT_THAT_I_OPENED=12345   # please choose a port where you can accept incoming tcp connections (or open that port if you're on a cloud)

export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT_THAT_I_OPENED
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT_THAT_I_OPENED
export CUDA_VISIBLE_DEVICES=  # do not use GPUs even if they are avilable
  
export INITIAL_PEERS=""
# ^-- space-separated initial peers from your experiment

# organizations
export WANDB_ENTITY=CALM
export HF_ORGANIZATION_NAME=CALM

# experiment name
export EXP_NAME=CALM
export WANDB_PROJECT=$EXP_NAME
export HF_MODEL_NAME=$EXP_NAME

export WANDB_API_KEY=TODO_get_your_wandb_key_here_wandb.ai/authorize
export HF_USER_ACCESS_TOKEN=TODO_create_user_access_token_here_with_WRITE_permissions_https://huggingface.co/settings/token
# note: you can avoid setting the two tokens above: in that case, the script will ask you to login to wandb and huggingface
  
# activate your anaconda environment
source ~/anaconda3/bin/activate

export WANDB_START_METHOD=thread
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT_THAT_I_OPENED
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT_THAT_I_OPENED
export CUDA_VISIBLE_DEVICES=

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

python run_aux_peer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --wandb_project $WANDB_PROJECT --identity ./identity --store_checkpoints --upload_interval 43200 --repo_url $HF_ORGANIZATION_NAME/$HF_MODEL_NAME --authorize --assist_in_averaging --bandwidth $BANDWIDTH
# Optionally, add more peers to the training via `--initial_peers ONE_OR_MORE PEERS_HERE`
```

If everything went right, it will print its address as such:
![image](https://user-images.githubusercontent.com/3491902/146950956-0ea06e77-15b4-423f-aeaa-02eb6aec06db.png)

Please copy this address and use it as ``--initial_peers`` with GPU/TPU trainers and other auxiliary peers.
</details>


<details>
  <summary><b>3. Setting up a trainer</b></summary>
Trainers are peers with GPUs (or other compute accelerators) that compute gradients, average them via all-reduce and perform optimizer steps.
There are two broad types of trainers: normal (full) peers and client mode peers. Client peers rely on others to average their gradients, but otherwise behave same as full peers. You can designate your trainer as a client-only using the `--client_mode` flag.
  
__When do I need client mode?__ if a peer is unreliable (e.g. will likely be gone in 1 hour) OR sits behind a firewall that blocks incoming connections OR has very unstable internet connection, it should be a client. For instance, it is recommended to set colab / kaggle peers as clients. In turn, cloud GPUs (even spot instances!) are generally more reliable and should be full peers.

Participating as a client is easy, you can find the code for that in **this colab notebook(TODO)**. Setting up a full peer is more difficult,
### Set up environment:

This part is the same as in auxiliary peer, except we don't need LFS (that was needed to upload checkpoints).
```bash
sudo apt install -y git tmux
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh > Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p ~/anaconda3
source ~/anaconda3/bin/activate
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

git clone https://github.com/NCAI-Research/CALM/
pip install https://github.com/learning-at-home/hivemind/archive/calm.zip
cd CALM && pip install -q -r requirements.txt &> log

# re-install bitsandbytes for the actual CUDA version
pip uninstall -y bitsandbytes-cuda111
pip install -y bitsandbytes-cuda113==0.26.0
  
# note: we use bitsandbytes for 8-bit LAMB, and in turn, bitsandbytes needs cuda -- even if you run on a non-CUDA device.
```

```bash
export MY_IP=`curl --ipv4 -s http://whatismyip.akamai.com/`
export PORT_THAT_I_OPENED=31337  # same requirements as for aux peer
export LISTEN_ON=/ip4/0.0.0.0/tcp/$PORT_THAT_I_OPENED
export ANNOUNCE_ON=/ip4/$MY_IP/tcp/$PORT_THAT_I_OPENED
export CUDA_VISIBLE_DEVICES=0  # supports multiple cuda devices!

# organization & experiment name
export WANDB_ENTITY=CALM
export HF_ORGANIZATION_NAME=CALM
export EXP_NAME=CALM
export WANDB_PROJECT=$EXP_NAME-hivemind-trainers
export HF_MODEL_NAME=$EXP_NAME

export WANDB_API_KEY=TODO_get_your_wandb_key_here_https://wandb.ai/authorize_OR_just_login_on_wandb
export HF_USER_ACCESS_TOKEN=TODO_create_user_access_token_here_with_WRITE_permissions_https://huggingface.co/settings/token
# note: you can avoid setting the two tokens above: in that case, the script will ask you to login to wandb and huggingface

export INITIAL_PEERS="/ip4/34.124.232.172/tcp/12345/p2p/QmdGDSzDEi7uo8pTGG7n8s2dW12VGoPQKiDVDoQaVAo3bf /ip4/193.106.95.184/tcp/12345/p2p/QmRgdEXySu8hEB3xUxexJPxcv7M41PggRDnUTf9kStdgup"
# ^-- If you're runnnng an indepent experiment, this must be your own initial peers. Can be either auxiliary peers or full gpu peers.


curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -  --json > speedtest.json
export BANDWIDTH=`python -c "import json; speedtest = json.load(open('speedtest.json')); print(int(max(1, min(speedtest['upload'], speedtest['download']) / 1e6)))"`
echo "Internet Bandwidth (Mb/s) = $BANDWIDTH"

ulimit -n 16384 # this line is important, ignoring it may cause Too Many Open Files

python run_trainer.py --run_id $EXP_NAME --host_maddrs $LISTEN_ON --announce_maddrs $ANNOUNCE_ON --initial_peers $INITIAL_PEERS --bandwidth $BANDWIDTH \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1
# you can tune per_device_train_batch_size, gradient_accumulation steps, --fp16, --gradient_checkpoints based on the device. A good rule of thumb is that the device should compute (batch size x num accumulations) gradients over 1-10 seconds. Setting very large gradient_accumulation_steps can cause your peer to miss an averaging round.

```
  
  
</details>

<details>
  <summary><b>Best (and worst) practices (TBA)</b></summary>
  TODO

- full redundancy, three instances of everything
- client-to-averager ratio
- gradient checkpointing
- multiple GPUs per peer
- if aux peers have less ram, you can assign it to only parts of functionality, e.g. disable --upload_interval
</details>

