import os
import joblib
import torch
import json

class Versioning():
    def __init__(self, model_dir="models/"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok = True)

    def save_xgb(self, model, metadata: dict, version: str):
        joblib.dump(model, f"{self.model_dir}XGB{version}.pkl")
        with open(f"{self.model_dir}XGB{version}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_xgb(self, version:str):
        model = joblib.load(f"{self.model_dir}XGB{version}.pkl")
        metadata = json.load(f"{self.model_dir}XGB{version}_meta.json", "r")
        return (model, metadata)
    
    def save_lstm(self, model, metadata:dict, version:str):
        torch.save(model.state_dict(), f"{self.model_dir}LSTM{version}.pt")
        with open(f"{self.model_dir}LSTM{version}_meta.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def load_lstm(self, model_class, input_size:int, version:str):
        model = model_class(input_size=input_size)
        model.load_state_dict(torch.load(f"{self.model_dir}LSTM{version}.pt"))
        metadata = json.load(f"{self.model_dir}LSTM{version}_meta.json", "r")
        model.eval()
        return (model, metadata)
    
    def list_versions(self):
        files = os.listdir(self.model_dir)
        meta_files = [f for f in files if f.endswith('_meta.json')]
        lstm = [f for f in meta_files if f.startswith('LSTM')]
        xgb = [f for f in meta_files if f.startswith('XGB')]

        for x, l in zip(xgb, lstm):

            path_XGB = os.path.join(self.model_dir, x)
            path_LSTM = os.path.join(self.model_dir, l)

        with open(path_XGB, 'r') as x:
            metadata_xgb = json.load(x)
            
        version = metadata_xgb.get("version", "Unknown")
        trained_at = metadata_xgb.get("trained_at", "Unknown")
        xgb_acc = metadata_xgb.get("xgb_directional_acc", "N/A")


        with open(path_LSTM, 'r') as l:
            metadata_lstm = json.load(l)
            
        version = metadata_lstm.get("version", "Unknown")
        trained_at = metadata_lstm.get("trained_at", "Unknown")
        lstm_acc = metadata_lstm.get("lstm_directional_acc", "N/A")
        
        print(f"Version: {version} | Trained at: {trained_at}")
        print(f"  -> XGBoost Dir Accuracy: {xgb_acc:.2%}" if isinstance(xgb_acc, float) else f"  -> XGBoost Dir Accuracy: {xgb_acc}")
        print(f"  -> LSTM Dir Accuracy:    {lstm_acc:.2%}" if isinstance(lstm_acc, float) else f"  -> LSTM Dir Accuracy:    {lstm_acc}")   

