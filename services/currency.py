# backend/services/currency.py

from typing import Dict

# Versão inicial: tabela estática.
# Depois você pode trocar para ler de uma tabela SUPABASE: "exchange_rates"
DEFAULT_RATES = {
    "EUR": 1.0,      # moeda de referência
    "BRL": 0.18,     # 1 BRL = 0.18 EUR (exemplo!)
    "USD": 0.93,     # 1 USD = 0.93 EUR (exemplo!)
}

def get_rates_from_db_or_default() -> Dict[str, float]:
    """
    TODO: aqui você pode:
     - ler do Supabase uma tabela exchange_rates (currency, rate_to_eur)
     - ou chamar uma API externa
    Por enquanto, devolvemos estático.
    """
    return DEFAULT_RATES

def convert_amount(amount: float, from_currency: str, to_currency: str, rates: Dict[str, float], pivot="EUR") -> float:
    if from_currency == to_currency:
        return round(float(amount), 2)

    if from_currency not in rates or to_currency not in rates:
        # se alguma moeda faltar, devolve o valor original
        return round(float(amount), 2)

    # 1. converte 'from' -> pivot
    #  se rates[x] = valor em pivot de 1 unidade de x
    amount_in_pivot = float(amount) * rates[from_currency]

    # 2. converte pivot -> 'to'
    amount_in_target = amount_in_pivot / rates[to_currency]

    return round(amount_in_target, 2)
