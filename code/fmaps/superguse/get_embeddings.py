import os
import glob
from itertools import combinations as cb
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.colors as mc
# import pyrsa
# import pyrsa.data as rsd  # abbreviation to deal with dataset
# from pyrsa.vis.colors import rdm_colormap


def get_google_encoder(model_dir=None):
    """[fetch the google universal sentence encoder model from the internets]

    Args:
        model_dir ([os.path]): path where model will find shelter.

    Returns:
        model: tensorflow graph of guse
    """
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

    if model_dir is None:
        model = hub.load(module_url)
    else:
        export_module_dir = os.path.join(model_dir, "google_encoding")
        if not os.path.exists(export_module_dir):
            # dl and save
            model = hub.load(module_url)
            tf.saved_model.save(model, export_module_dir)
            print("module %s loaded" % module_url)
        else:
            # load the pre-dl model
            model = hub.load(export_module_dir)
            print("module %s loaded" % export_module_dir)

    return model
    
def get_mpnet_encoder(model_dir=None):
    """[fetch the google universal sentence encoder model from the internets]

    Args:
        model_dir ([os.path]): path where model will find shelter.

    Returns:
        model: tensorflow graph of guse
    """
    # from transformers import MPNetModel, MPNetConfig
    # configuration = MPNetConfig()
    # model = MPNetModel(configuration)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    return model


def get_guse_data(datapath, model=None):
    """load the sentence captions from X participants
       that captioned the 240 stimuli on Meadows-research.com

    Args:
        datapath (os.path): path where the data lives. 'data'
        case ('collate', 'indep')

    Returns:
        data (list): list of sentences, sorting according to
                     stim1_name
    """
    if model is None:
        model = get_google_encoder()
    datafiles = glob.glob(os.path.join(datapath, '*.csv'))
    datafiles.sort()
    datas = []
    for file in datafiles:
        data = pd.read_csv(file)
        # Sort the rows of dataframe by stimulus name
        rs_stimnames = np.unique(
            np.asarray(
                data.sort_values(by='stim1_name')['stim1_name']
                )
            )
        rs_sentences = np.asarray(
            data.sort_values(by='stim1_name')['label']
            )
        datas.append(rs_sentences)

    # here we either collate the sentences in a paragraph per
    # stimulus (to account for individual differences)
    embeddings = []
    rdms = []
    for sub_i, data in enumerate(datas):
        print(sub_i)
        embedding = model(data)
        nCond, nDim = embedding.shape

        # create the descriptors for pyrsa
        des = {'subj': sub_i + 1}
        obs_des = {'conds': rs_stimnames}

        chn_des = {
            'dimensions': np.array(
                [f'dimension_{x+1}' for x in np.arange(nDim)]
                )
            }

        # create a Dataset
        data = np.asarray(embedding)
        # rsd.Dataset(
        #     measurements=np.asarray(embedding),
        #     descriptors=des,
        #     obs_descriptors=obs_des,
        #     channel_descriptors=chn_des)
        print(data)

        embeddings.append(data)

    # now we collected the single subject datasets, let's compute the RDMs
    # rdms = pyrsa.rdm.calc_rdm(embeddings, method='correlation')

    return embeddings, rs_stimnames, datas



def get_mpnet_data(datapath, model=None):
    """load the sentence captions from X participants
       that captioned the 240 stimuli on Meadows-research.com

    Args:
        datapath (os.path): path where the data lives. 'data'
        case ('collate', 'indep')

    Returns:
        data (list): list of sentences, sorting according to
                     stim1_name
    """
    if model is None:
        model = get_mpnet_encoder()
    datafiles = glob.glob(os.path.join(datapath, '*.csv'))
    # datafiles   = datapath
    # if datafiles

    
    for file in datafiles:
        # print(file)
        data = pd.read_csv(file)

        datas_stim = data['img_names']
        datas_sent = data['gen_texts']

    # here we either collate the sentences in a paragraph per
    # stimulus (to account for individual differences)
    embeddings = []
    for file, data in enumerate(datas_sent):
        # print(file)
        embedding = model.encode(data)
        # nCond, nDim = embedding.shape

        # # create the descriptors for pyrsa
        # des = {'subj': file + 1}
        # obs_des = {'conds': rs_stimnames}

        # chn_des = {
        #     'dimensions': np.array(
        #         [f'dimension_{x+1}' for x in np.arange(nDim)]
        #         )
        #     }

        # create a Dataset
        data_embed = np.asarray(embedding)
        # rsd.Dataset(
        #     measurements=np.asarray(embedding),
        #     descriptors=des,
        #     obs_descriptors=obs_des,
        #     channel_descriptors=chn_des)
        # print(data_embed)

        embeddings.append(data_embed)

    # now we collected the single subject datasets, let's compute the RDMs
    # rdms = pyrsa.rdm.calc_rdm(embeddings, method='correlation')

    return embeddings, datas_stim, datas_sent
