from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import utils


thresh=0.60
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name='saved_model/saved'


model_sim = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)



def predict_reason(df):
    batch = tokenizer(df['text'].values[0], truncation=True, padding='longest', return_tensors="pt").to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def compare_reason(true_reason,predicted_reason,thresh):
    sentences = [true_reason, predicted_reason]
    embedding_1= model_sim.encode(sentences[0], convert_to_tensor=True)
    embedding_2 = model_sim.encode(sentences[1], convert_to_tensor=True)
    val=util.pytorch_cos_sim(embedding_1, embedding_2)
    if(val.cpu().detach().numpy()[0][0]>=thresh):
        return 1
    return 0
    
def predict(text,reason):
    dict_df={'text':[text],'reason':[reason]}
    df=pd.DataFrame(dict_df)
    df=utils.clean_data(df)
    predicted_reason=predict_reason(df)
    return compare_reason(df["reason"].values[0],predicted_reason,thresh),predicted_reason

def predict_test(df):
    print("model name")
    dict_df={"Id":[],"Abstract":[],"RHS":[]}
    print("model loaded")
    for i in range(df.shape[0]):
        vl=df.iloc[i].values
        dict_df["FileName"].append(vl[0])
        dict_df["Abstract"].append(vl[1])
        batch = tokenizer(vl[1], truncation=True, padding='longest', return_tensors="pt").to(device)
        translated = model.generate(**batch)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        dict_df["RHS"].append(tgt_text)

    res=pd.DataFrame(dict_df)
    return res