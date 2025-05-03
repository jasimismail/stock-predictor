from django.contrib import admin
from .models import PredictionResult

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('stock_symbol', 'signal', 'confidence', 'model_type', 'timestamp')
    list_filter = ('model_type', 'signal', 'stock_symbol')
    search_fields = ('stock_symbol',)
    date_hierarchy = 'timestamp'
