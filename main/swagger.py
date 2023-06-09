from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from django.urls import path
from rest_framework import permissions

schema_view = get_schema_view(
    openapi.Info(
        title="Clustering Service API",
        default_version='v1',
        description="Сервис тематической кластеризации научных документов цифровой математической библиотеки",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="nikita.shirmanov@bk.ru"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('swagger(<format>\.json|\.yaml)', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
