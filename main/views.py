import codecs
import csv
import numpy as np


import requests
from bs4 import BeautifulSoup


import torch
from django.http import HttpResponse
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from hdbscan import HDBSCAN
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.metrics import davies_bouldin_score
from transformers import BertModel, BertTokenizer


def process_parse(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, features="html.parser")
    issues = soup.find_all('td')
    issues_links = []
    for issue in issues:
        parse_links = issue.find_all('a')
        for parse_link in parse_links:
            issues_links.append(parse_link.get('href'))

    titles = []
    for issue_link in issues_links:
        url = 'https:' + issue_link
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features='html.parser')
        articles = soup.find_all('div', class_='articleItem')
        for article in articles:
            print()
            titles.append(article.div.a.text)
    return titles


def generate_csv_file(articles):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="articles.csv"'

    writer = csv.writer(response)
    writer.writerow(['title'])
    for article in articles:
        writer.writerow([article])
    return response


class TextParsingView(APIView):

    @swagger_auto_schema(
        method='post',
        operation_summary="Парсинг статей",
        operation_description="Парсинг статей и их сохранение в csv файл.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'link': openapi.Schema(type=openapi.TYPE_STRING)
            }
        ),
        responses={200: "OK"}
    )
    @action(methods=['post'], detail=False)
    def post(self, request):
        return generate_csv_file(process_parse(request.data.get("link")))


class TextClusteringView(APIView):
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        operation_summary="Кластеризация статей",
        operation_description="Кластеризует статьи на основе их заголовков с помощью алгоритма HDBSCAN и BERT-эмбеддингов.",
        consumes=["multipart/form-data"],
        responses={200: "OK"},
        manual_parameters=[
            openapi.Parameter(
                'file',  # имя параметра
                in_=openapi.IN_FORM,  # указываем, что параметр передается в форме
                type=openapi.TYPE_FILE,  # указываем, что параметр типа file
                required=True,  # делаем параметр обязательным
                description='CSV файл с заголовками статей'  # описание параметра
            )
        ]
    )
    def post(self, request, *args, **kwargs):

        # Получаем путь к csv файлу из параметров запроса
        csv_file = self.request.FILES.get('file')

        # Читаем csv файл и сохраняем заголовки в список titles
        documents = []
        reader = csv.reader(codecs.iterdecode(csv_file, 'utf-8'))
        next(reader)  # пропускаем заголовок таблицы
        for row in reader:
            documents.append(row[0])

        # ансамблирования моделей Bert
        models = [BertModel.from_pretrained('sberbank-ai/ruBert-large'),
                  BertModel.from_pretrained('sberbank-ai/ruBert-base')]

        # создаем токенизатор
        tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-large')

        # список для хранения эмбеддингов
        embeddings = []

        # получение эмбеддинга для каждого документа от каждой модели и объединение их
        for text in documents:
            input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
            with torch.no_grad():
                text_embeddings = []
                for model in models:
                    last_hidden_states = model(input_ids)[0]
                    text_embedding = np.mean(last_hidden_states.numpy(), axis=1)
                    text_embeddings.append(text_embedding)
                text_embeddings = np.concatenate(text_embeddings, axis=1)
                embeddings.append(text_embeddings)

        # объединение эмбеддингов для каждого документа в единый массив
        embeddings = np.vstack(embeddings)

        cluster = HDBSCAN(metric='euclidean', cluster_selection_method='eom')
        cluster_labels = cluster.fit_predict(embeddings)

        # Вычисление метрики Индекс Давида-Болдина
        davies_bouldin_index = davies_bouldin_score(embeddings, cluster_labels)
        print("Индекс Давида-Болдина: ", davies_bouldin_index)


        # создаем словарь с сопоставлением текстов и меток кластеров
        cluster_dict = {}
        for i, text in enumerate(documents):
            label = cluster_labels[i]
            if label in cluster_dict:
                cluster_dict[label]['documents'].append(text)
            else:
                cluster_dict[label] = {'documents': [text]}

        # создаем новый словарь для хранения измененных ключей
        new_cluster_dict = {}
        for label in cluster_dict:
            new_key = str(label)  # приводим метки кластеров к типу str
            new_cluster_dict[new_key] = cluster_dict[label]

        # возвращаем результаты в формате JSON
        result = {
            'clusters': new_cluster_dict,
            'davies_bouldin_index': davies_bouldin_index
        }

        return Response(result)
