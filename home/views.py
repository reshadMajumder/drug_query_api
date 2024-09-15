
import time
import pandas as pd
import torch
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel
from django.conf import settings

import openai
# pip install openai==0.28
import torch

# Set your API keys and index name
api_key = settings.PINECONE_API_KEY
index_name = 'storer'
dimension = 768  # Dimension for BERT embeddings

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Define serverless specifications for Pinecone
spec = ServerlessSpec(cloud='aws', region='us-east-1')

# Create the index if it does not exist
if index_name not in [index_info["name"] for index_info in pc.list_indexes()]:
    pc.create_index(name=index_name, dimension=dimension, metric='cosine', spec=spec)
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print(f"Index {index_name} has been successfully created.")

# Load BERT tokenizer and model for generating vectors
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Function to generate vectors from text
def generate_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Django REST API view to handle file uploads
@api_view(['POST'])
def upload_to_pinecone(request):
    file = request.FILES.get('file')

    if not file or not (file.name.endswith('.xlsx') or file.name.endswith('.csv')):
        return Response({"error": "Invalid file format. Please upload an Excel (.xlsx) or CSV file."}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Read the file into a DataFrame
        if file.name.endswith('.xlsx'):
            df = pd.read_excel(file, engine='openpyxl')
        else:
            df = pd.read_csv(file)

        # Replace empty cells with 'unavailable'
        df.fillna('unavailable', inplace=True)

        # Generate vectors for each row based on the 'Drug Name' column
        df['vector'] = df['Drug Name'].apply(generate_vector)

        # Ensure vectors have the correct dimension
        if len(df['vector'].iloc[0]) != dimension:
            return Response({"error": f"Vector dimension does not match index dimension: {dimension}"}, status=status.HTTP_400_BAD_REQUEST)

        # Split the DataFrame into chunks
        chunk_size = 100
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Connect to the Pinecone index
        index = pc.Index(index_name)

        # Upload the data chunk by chunk
        for chunk_num, chunk in enumerate(chunks):
            vectors = []
            for idx, row in chunk.iterrows():
                vector = row['vector']
                metadata = row.drop('vector').to_dict()  # Exclude the vector column for metadata
                vectors.append({
                    "id": str(idx),  # Ensure each ID is unique
                    "values": vector,
                    "metadata": metadata
                })

            # Upsert the vectors into the Pinecone index
            if vectors:
                index.upsert(vectors=vectors)
                print(f"Chunk {chunk_num + 1}/{len(chunks)} successfully uploaded.")
        
        return Response({"message": "All data successfully uploaded to Pinecone."}, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)








# ==================chat =====================




# Set your API keys and index name
pinecone_api_key = 'c1a5783a-5a44-44b9-8298-5d63bfabae0c'
openai_api_key = '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'
index_name = 'storer'

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Load tokenizer and model for embeddings
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Set up OpenAI API
openai.api_key = openai_api_key

def generate_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

def query_pinecone(query_text, top_k=1):
    query_vector = generate_vector(query_text)
    query_response = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return query_response['matches']

def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing information about drugs based on a database. Provide concise and informative responses using ONLY the information given. If the information doesn't fully answer the query, just provide what's available without speculation."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

@api_view(['POST'])
def chat(request):
    if 'question' not in request.data:
        return Response({"error": "No question provided"}, status=status.HTTP_400_BAD_REQUEST)

    user_input = request.data['question']
    
    # Query Pinecone
    results = query_pinecone(user_input)
    
    if not results:
        return Response({"response": "I'm sorry, but I couldn't find any information about that in my database. Could you please try asking about a different drug or rephrasing your question?"}, status=status.HTTP_404_NOT_FOUND)
    
    best_match = results[0]
    drug_info = format_drug_info(best_match['metadata'])
    
    # Prepare prompt for GPT
    prompt = f"Based on the following drug information:\n\n{drug_info}\n\nUser question: {user_input}\n\nPlease provide a helpful response using ONLY the information given:"
    
    # Get GPT response
    gpt_response = get_gpt_response(prompt)
    
    return Response({"response": gpt_response}, status=status.HTTP_200_OK)

def format_drug_info(drug_info):
    formatted = f"Drug Name: {drug_info.get('Drug Name', 'N/A')}\n"
    formatted += f"Vendor: {drug_info.get('Vendor', 'N/A')}\n"
    formatted += f"Price: {drug_info.get('Price', 'N/A')}\n"
    formatted += f"Quantity: {drug_info.get('Quantity', 'N/A')}\n"
    return formatted






















# # Create your views here.
# # myapp/views.py
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework.parsers import MultiPartParser
# import pandas as pd
# import pinecone
# import openai


# import os
# from pinecone import Pinecone, ServerlessSpec

# # Initialize Pinecone instance
# pc = Pinecone(
#     api_key="c1a5783a-5a44-44b9-8298-5d63bfabae0c"
# )

# # Create index if it doesn't exist
# if 'jojo' not in pc.list_indexes().names():
#     pc.create_index(
#         name='jojo',
#         dimension=1536,
#         metric='euclidean',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-west-2'
#         )
#     )

# # Your other view logic

# # Initialize Pinecone
# # pinecone.init(api_key="c1a5783a-5a44-44b9-8298-5d63bfabae0c", environment="https://jojo-gd9j5w5.svc.aped-4627-b74a.pinecone.io")
# # index = pinecone.Index("jojo")  # Create this index beforehand

# # OpenAI API key setup
# openai.api_key = "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"

# @api_view(['POST'])
# def upload_file(request):
#     file = request.FILES.get('file')
    
#     if not file or not (file.name.endswith('.xlsx') or file.name.endswith('.xls')):
#         return Response({"error": "Invalid file format. Please upload an Excel file."}, status=400)
    
#     try:
#         # Read Excel file using pandas
#         df = pd.read_excel(file)
        
#         # Store data in Pinecone
#         store_data_in_pinecone(df)

#         return Response({"message": "File uploaded and data stored successfully."}, status=200)
    
#     except Exception as e:
#         return Response({"error": str(e)}, status=500)

# def store_data_in_pinecone(dataframe):
#     for i, row in dataframe.iterrows():
#         # Create a vector for each row (this example uses OpenAI to generate embeddings)
#         vector = create_vector_from_row(row)
#         # Store vector in Pinecone with a unique ID
#         index.upsert(vectors=[(str(i), vector)])  

# def create_vector_from_row(row):
#     # Convert the row data to a single string (adjust this depending on your data)
#     data_string = ' '.join(map(str, row))
    
#     # Get vector representation from GPT-4 API (or use any embedding model)
#     response = openai.Embedding.create(input=data_string, model="text-embedding-ada-002")
#     vector = response['data'][0]['embedding']
    
#     return vector




# # myapp/views.py
# @api_view(['POST'])
# def query_data(request):
#     query_text = request.data.get('query', '')

#     if not query_text:
#         return Response({"error": "Query text is required"}, status=400)

#     try:
#         # Get vector representation for the query text
#         query_vector = get_query_embedding(query_text)
        
#         # Search for the closest vectors in Pinecone
#         search_results = index.query(queries=[query_vector], top_k=5)
        
#         # Collect relevant metadata from the search results
#         relevant_data = [match['metadata'] for match in search_results['matches']]
        
#         # Generate a response using GPT-4
#         gpt_response = get_gpt4_response(query_text, relevant_data)
        
#         return Response({"response": gpt_response}, status=200)
    
#     except Exception as e:
#         return Response({"error": str(e)}, status=500)

# def get_query_embedding(query_text):
#     # Get embedding for the query text using GPT-4
#     response = openai.Embedding.create(input=query_text, model="text-embedding-ada-002")
#     return response['data'][0]['embedding']

# def get_gpt4_response(query_text, relevant_data):
#     # Pass relevant data to GPT-4 along with the query to generate a response
#     context = "\n".join(relevant_data)
#     completion = openai.Completion.create(
#         engine="gpt-4",
#         prompt=f"Context:\n{context}\n\nQuery: {query_text}\nAnswer:",
#         max_tokens=150
#     )
#     return completion.choices[0].text.strip()
