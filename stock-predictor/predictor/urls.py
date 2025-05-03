from django.urls import path
from .views import (
    RandomForestView,
    LightGBMView,
    XGBoostView,
    CatBoostView,
    DashboardView
)

urlpatterns = [
    path('', DashboardView.as_view(), name='dashboard'),
    path('random-forest/', RandomForestView.as_view(), name='predict_random_forest'),
    path('lightgbm/', LightGBMView.as_view(), name='predict_lightgbm'),
    path('xgboost/', XGBoostView.as_view(), name='predict_xgboost'),
    path('catboost/', CatBoostView.as_view(), name='predict_catboost'),
]
