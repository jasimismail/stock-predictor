from django import forms

class StockPredictionForm(forms.Form):
    tickers = forms.CharField(
        label='Stock Symbols (comma-separated)',
        max_length=200,
        widget=forms.TextInput(attrs={
            'placeholder': 'RELIANCE.NS, TATASTEEL.NS, INFY.NS',
            'class': 'form-control'
        }),
        help_text='Enter stock symbols separated by commas'
    )
