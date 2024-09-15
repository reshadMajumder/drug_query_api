# myapp/urls.py
from django.urls import path
from .views import upload_to_pinecone,chat

urlpatterns = [
    path('upload', upload_to_pinecone, name='upload_to_pinecone'),
    path('chat', chat, name='chat'),

    # path('query', query_data, name='query_data'),
]
