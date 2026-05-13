import sys
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from time import time
import pickle
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
import chromadb
from chromadb.utils import embedding_functions


split1, split2 = 0, 500

splitstr = str(split1)+'_'+str(split2)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


################### Hyperparameters ###########################
dataframe_path = '/scratch/NLU/cvlachos/SCQA/Samu_XLSR_finetuning/data/step4_fixed_props_asr.parquet'
output_dir = '/scratch/NLU/cvlachos/SCQA/Models_of_Samu_XLSR_finetuning/fine_tuned_text_implicit_model_doc2dialsplit_'+splitstr
batch_size = 1

df = pd.read_parquet(dataframe_path)

if not os.path.exists(output_dir) : os.makedirs(output_dir)

######################## Model definition

model = SentenceTransformer('sentence-transformers/LaBSE').to(device)
labse_encoder = SentenceTransformer('sentence-transformers/LaBSE').to(device)
for param in labse_encoder.parameters():
    param.requires_grad = False

tokenizer = labse_encoder.tokenizer
tokenizer.truncation_side = 'left'
model.tokenizer.truncation_side = 'left'



################### Load Data ###########################
# df = pd.read_parquet(dataframe_path)[:1]

# class Samu_Dataset(Dataset):
#     def __init__(self, df):
#         self.df = df
#     def __len__(self):
#         return self.df.shape[0]
#     def __getitem__(self, idx):        
#         return df.iloc[idx]


# def data_collator(batch):
#     rewrite_list = []
#     input_txt = []

#     for row in batch:
#         rewrite_list.append(row['rewrite'])
#         input_txt.append(" [SEP] ".join(row['context_qu_txt']))


#     return {'input_txt':input_txt, 'labels':rewrite_list}



# if split2 == -1: split2=df.shape[0]
# df = df.iloc[split1:split2]
# test_dataset = Samu_Dataset(df)

# test_dataloader = DataLoader(
#     dataset=test_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     collate_fn=data_collator
# )

print('Spliting dataset. Test contains indexes',split1,'-',split2)




################################### Evaluation/testing
def evaluate(test_loader):
    model.eval()
    total_loss = 0.0
    start = time()
    embeds = []

    with torch.no_grad():
        for nn, batch in enumerate(test_loader):
            print('Encoding',nn,'//',len(test_loader))
            print(batch['input_txt'])

            features = model.tokenize(batch['input_txt'])
            features = batch_to_device(features, device)
            m_res = model(features)["sentence_embedding"]  


            embeds.append(m_res.detach().cpu().squeeze().numpy())

    print("Evaluation time:", time() - start)
    return embeds

def generate_embedding(text):
    model.eval()
    total_loss = 0.0
    start = time()
    embeds = []

    with torch.no_grad():
        print(text)

        features = model.tokenize(text)
        features = batch_to_device(features, device)
        m_res = model(features)["sentence_embedding"]  


        embeds.append(m_res.detach().cpu().squeeze().numpy())

    print("Evaluation time:", time() - start)
    return embeds


####################################### Training

if "best.pt" in os.listdir(output_dir):
  checkpoint = torch.load(os.path.join(output_dir, 'best.pt'), map_location=torch.device(device))
  model.load_state_dict(checkpoint['state_dict'])

text_list = ['Hello', 'Hello! How can I assist you today?', 'What are supplementary policy benefits and are they included in a conversion policy?']
text = [" [SEP] ".join(text_list)]


embeds_2 = generate_embedding(text)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/LaBSE"
)

client = chromadb.PersistentClient(path="/scratch/NLU/cvlachos/SCQA/Samu_XLSR_finetuning/data/chroma_db")
collection = client.get_or_create_collection(
    name="propositions_VS",
    embedding_function=sentence_transformer_ef
)


query_embedding = [np.array(embeds_2)]

query_embedding = query_embedding[0].squeeze()


results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

print(results)