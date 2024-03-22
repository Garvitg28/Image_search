# Import necessary libraries
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import chromadb
import re
import numpy as np

# Set the device to CPU
device = 'cpu'  

# CLIP Model Helper class
class CLIPModelHelper:
    def __init__(self, model_ID, device):
        # Initialize the CLIP model, processor, and tokenizer
        self.model, self.processor, self.tokenizer = self.get_model_info(model_ID, device)

    def get_model_info(self, model_ID, device):
        # Load the CLIP model, processor, and tokenizer
        model = CLIPModel.from_pretrained(model_ID).to(device)
        processor = CLIPProcessor.from_pretrained(model_ID)
        tokenizer = CLIPTokenizer.from_pretrained(model_ID)

        return model, processor, tokenizer

    def get_single_text_embedding(self, text, device):
        # Get the CLIP embedding for a single text input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        text_embeddings = self.model.get_text_features(**inputs)
        embedding_as_np = text_embeddings.cpu().detach().numpy()

        return embedding_as_np

    def get_all_text_embeddings(self, df, text_col, device):
        # Get CLIP embeddings for all text inputs in a DataFrame
        df["text_embeddings"] = df[str(text_col)].apply(self.get_single_text_embedding)

        return df

    def get_single_image_embedding(self, my_image, device):
        # Get the CLIP embedding for a single image input
        image = self.processor(text=None, images=my_image, return_tensors="pt")["pixel_values"].to(device)
        embedding = self.model.get_image_features(image)
        embedding_as_np = embedding.cpu().detach().numpy()

        return embedding_as_np

    def get_all_images_embedding(self, df, img_column, device):
        # Get CLIP embeddings for all image inputs in a DataFrame
        df["img_embeddings"] = df[str(img_column)].apply(lambda x: self.get_single_image_embedding(x, device))

        return df

# Data Helper class
class DataHelper:
    @staticmethod
    def convert_string_to_array(input_string):
        # Convert a string representation to a NumPy array
        numeric_values = re.findall(r'-?\d+\.\d+e[+-]?\d+', input_string)
        numpy_array = np.array(numeric_values, dtype=np.float32).reshape(1, -1)
        formatted_string = ', '.join(map(str, numpy_array[0]))

        return f'[{formatted_string}]'

    @staticmethod
    def convert_to_numpy_array(df):
        # Convert DataFrame columns to NumPy arrays
        df = df.apply(DataHelper.convert_string_to_array)
        column_as_numpy = df.apply(lambda x: np.array(eval(x)))
        column_reshaped = column_as_numpy.apply(lambda x: x.reshape(-1, 768))

        return column_reshaped

# ChromaDB Manager class
class ChromaDBManager:
    def __init__(self, host, port):
        # Initialize the ChromaDB HTTP client and collection
        self.client = chromadb.HttpClient(host=host, port=port)
        self.collection = None

    def initialize_collection(self, collection_name):
        # Initialize the collection in ChromaDB
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_data_to_collection(self, images_df, df, metadatas, img_embedding_column='flattened_embeddings', id_column='id'):
        # Add data to the ChromaDB collection
        embeddings = images_df[img_embedding_column].tolist()
        ids_list = df 
        self.collection.add(embeddings=embeddings, ids=ids_list, metadatas=metadatas)

    def change_metadata_from_collection(self, text_df, images_df, df_change, metadatas_change, text_column='text', img_embedding_column='flattened_embeddings', id_column='id'):
        # Change metadata in the ChromaDB collection
        self.collection.delete(ids=df_change)     
        documents = text_df[text_column].tolist()
        embeddings = images_df[img_embedding_column].tolist()
        ids = df_change
        self.collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas_change)

    def delete_from_collection(self, df_del):
        # Delete entries from the ChromaDB collection
        self.collection.delete(ids=df_del)

    def get_from_collection(self, df_get):
        # Retrieve entries from the ChromaDB collection
        res = self.collection.get(ids=df_get)
        return res

    def query_collection(self, query_emb, n_results):
        # Query the ChromaDB collection based on specified query embeddings
        results = self.collection.query(query_embeddings=query_emb, n_results=n_results)
        return results
