"""
Views for the stock predictor application.
"""

from django.shortcuts import render, redirect
from django.views import View
from django.contrib import messages
from django.urls import resolve
from django.utils import timezone
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .models import PredictionResult
from .forms import StockPredictionForm
from .core.services.prediction_service import PredictionService

import logging

logger = logging.getLogger('stock_predictor')

class BaseModelView(View):
    """
    Base view for model-specific prediction pages.
    """
    template_name = None
    model_type = None
    model_name = None
    
    def get(self, request):
        if not all([self.template_name, self.model_type, self.model_name]):
            logger.error("View not properly configured: missing template_name, model_type, or model_name")
            messages.error(request, "View configuration error")
            return redirect('dashboard')
            
        form = StockPredictionForm()
        try:
            # Get recent predictions for this model type
            predictions = PredictionResult.objects.filter(
                model_type=self.model_type
            ).order_by('-timestamp')[:10]
            
            # Calculate accuracy for the last 30 days
            end_date = timezone.now()
            start_date = end_date - timedelta(days=30)
            
            accuracy_data = PredictionResult.objects.filter(
                model_type=self.model_type,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            )
            
            total_predictions = accuracy_data.count()
            correct_predictions = accuracy_data.filter(signal='Buy').count()
            
            accuracy = {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
            }
            
        except Exception as e:
            predictions = []
            accuracy = {}
            messages.warning(request, "Database not ready. Please run migrations first.")
            logger.error(f"Database error: {str(e)}")
            
        return render(request, self.template_name, {
            'form': form,
            'predictions': predictions,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'accuracy': accuracy
        })
    
    def post(self, request):
        form = StockPredictionForm(request.POST)
        if form.is_valid():
            try:
                tickers_input = form.cleaned_data['tickers']
                tickers = [t.strip() for t in tickers_input.split(',') if t.strip()]
                
                if not tickers:
                    messages.error(request, "Please enter at least one valid stock symbol")
                    return redirect(f'predict_{self.model_type}')
                
                logger.info(f"Processing prediction request for tickers: {tickers} using {self.model_type} model")
                
                # Create prediction service
                service = PredictionService(model_type=self.model_type)
                
                # Make predictions
                results = service.predict_stocks(tickers)
                
                # Process results
                success_count = 0
                for ticker, result in results.items():
                    if result.get('signal') == 'Error':
                        error_msg = result.get('error', 'Unknown error')
                        logger.error(f"Error predicting {ticker}: {error_msg}")
                        messages.error(request, f"Error predicting {ticker}: {error_msg}")
                    else:
                        signal = result.get('signal', 'Unknown')
                        confidence = result.get('confidence', 0.0)
                        logger.info(f"Successful prediction for {ticker}: {signal} ({confidence:.2%})")
                        messages.success(request, f"Prediction for {ticker}: {signal} ({confidence:.2%})")
                        success_count += 1
              
                if success_count > 0:
                    messages.info(request, f"Successfully processed {success_count} out of {len(tickers)} stocks")
              
            except Exception as e:
                logger.error(f"Error processing prediction request: {str(e)}", exc_info=True)
                messages.error(request, f"Error processing prediction request: {str(e)}")
              
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"Error in {field}: {error}")
                  
        return redirect(f'predict_{self.model_type}')

class RandomForestView(BaseModelView):
    """
    View for Random Forest predictions.
    """
    template_name = 'predictor/random_forest.html'
    model_type = 'random_forest'
    model_name = 'Random Forest'

class LightGBMView(BaseModelView):
    """
    View for LightGBM predictions.
    """
    template_name = 'predictor/lightgbm.html'
    model_type = 'lightgbm'
    model_name = 'LightGBM'

class XGBoostView(BaseModelView):
    """
    View for XGBoost predictions.
    """
    template_name = 'predictor/xgboost.html'
    model_type = 'xgboost'
    model_name = 'XGBoost'

class CatBoostView(BaseModelView):
    """
    View for CatBoost predictions.
    """
    template_name = 'predictor/catboost.html'
    model_type = 'catboost'
    model_name = 'CatBoost'

class DashboardView(View):
    """
    View for the main dashboard.
    """
    template_name = 'predictor/dashboard.html'
    
    def get(self, request):
        try:
            all_predictions = {}
            all_accuracy = {}
            
            # Get predictions and accuracy for each model type
            for model_type in ['random_forest', 'lightgbm', 'xgboost', 'catboost']:
                # Get recent predictions
                predictions = PredictionResult.objects.filter(
                    model_type=model_type
                ).order_by('-timestamp')[:5]
                
                # Calculate accuracy for the last 30 days
                end_date = timezone.now()
                start_date = end_date - timedelta(days=30)
                
                accuracy_data = PredictionResult.objects.filter(
                    model_type=model_type,
                    timestamp__gte=start_date,
                    timestamp__lte=end_date
                )
                
                total_predictions = accuracy_data.count()
                correct_predictions = accuracy_data.filter(signal='Buy').count()
                
                all_predictions[model_type] = predictions
                all_accuracy[model_type] = {
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions,
                    'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0
                }
                
        except Exception as e:
            all_predictions = {}
            all_accuracy = {}
            messages.warning(request, "Database not ready. Please run migrations first.")
            logger.error(f"Database error: {str(e)}")
            
        return render(request, self.template_name, {
            'predictions': all_predictions,
            'accuracy': all_accuracy
        })
