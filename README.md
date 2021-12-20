# CALM: Collaborative Arabic Language Model
The CALM project is joint effort lead by NCAI in collaboration with Yandex and HuggingFace to train an Arabic language model with volunteers from around the globe. The project is an adaptation of the framework proposed at the NeurIPS 2021 demonstration: Training Transformers Together. 


Once of the main obstacles facing many researchers in the Arabic NLP community is the lack of computing resources that are needed for training large models. Models with leading performane on Arabic NLP tasks, such as AraBERT, CamelBERT, AraELECTRA, and QARiB, took days to train on TPUs. In the spirit of democratization of AI and community enabling, a core value at NCAI, CALM aims to demonstrate the effectiveness of collaborative training and form a community of volunteers for ANLP researchers with basic level cloud GPUs who wish to train their own models collaboratively. 

CALM trains a single BERT model on a dataset that combines MSA, Oscar and Arabic Wikipedia, and dialectal data for the gulf region from existing open source datasets. Each volunteer GPU trains the model locally at its own pace on a portion of the dataset while another portion is being streamed in the background to reduces local memory consumption. Computing the gradients and aggregating them is performed in a distributed manner, based on the computing abilities of each participating volunteer. Details of the distributed training process are further described here.

