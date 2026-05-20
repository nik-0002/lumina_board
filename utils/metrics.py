from typing import Dict, List
import numpy as np

class MetricsCalculator:
    """
    Calculate agricultural marketing metrics and KPIs
    """
    
    @staticmethod
    def calculate_delivery_rate(delivered: int, sent: int) -> float:
        """Calculate message delivery rate (%)"""
        return (delivered / sent * 100) if sent > 0 else 0
    
    @staticmethod
    def calculate_engagement_rate(engaged: int, delivered: int) -> float:
        """Calculate farmer engagement rate (%)"""
        return (engaged / delivered * 100) if delivered > 0 else 0
    
    @staticmethod
    def calculate_conversion_rate(converted: int, engaged: int) -> float:
        """Calculate conversion rate from engagement (%)"""
        return (converted / engaged * 100) if engaged > 0 else 0
    
    @staticmethod
    def calculate_roas(revenue: float, cost: float) -> float:
        """Calculate Return on Ad Spend"""
        return (revenue / cost) if cost > 0 else 0
    
    @staticmethod
    def calculate_cac(cost: float, conversions: int) -> float:
        """Calculate Customer Acquisition Cost"""
        return (cost / conversions) if conversions > 0 else 0
    
    @staticmethod
    def calculate_ltv(average_purchase_value: float, purchase_frequency: float,
                     customer_lifespan_years: float) -> float:
        """Calculate Lifetime Value of a farmer customer"""
        return average_purchase_value * purchase_frequency * (customer_lifespan_years * 12)
    
    @staticmethod
    def calculate_attribution_scores(interaction_sequence: List[Dict]) -> Dict[str, float]:
        """
        Calculate attribution scores for multi-touch journeys
        Supports: First-touch, Last-touch, Linear, Time-decay
        """
        if not interaction_sequence:
            return {}
        
        n = len(interaction_sequence)
        scores = {}
        
        # Linear attribution
        linear_weight = 1 / n
        
        # Time decay (exponential)
        decay_weights = np.exp(np.linspace(0, 1, n))
        decay_weights = decay_weights / decay_weights.sum()
        
        for i, interaction in enumerate(interaction_sequence):
            channel = interaction.get('channel', 'unknown')
            
            if channel not in scores:
                scores[channel] = {
                    'first_touch': 0,
                    'last_touch': 0,
                    'linear': 0,
                    'time_decay': 0
                }
            
            # First touch
            if i == 0:
                scores[channel]['first_touch'] = 1.0
            
            # Last touch
            if i == n - 1:
                scores[channel]['last_touch'] = 1.0
            
            # Linear
            scores[channel]['linear'] += linear_weight
            
            # Time decay
            scores[channel]['time_decay'] += decay_weights[i]
        
        return scores
    
    @staticmethod
    def calculate_trend(values: List[float], period: int = 7) -> str:
        """
        Determine trend direction (UP, DOWN, STABLE)
        """
        if len(values) < period:
            return "INSUFFICIENT_DATA"
        
        recent = np.mean(values[-period:])
        previous = np.mean(values[-2*period:-period])
        
        change_percent = ((recent - previous) / previous * 100) if previous != 0 else 0
        
        if change_percent > 5:
            return "UP"
        elif change_percent < -5:
            return "DOWN"
        else:
            return "STABLE"
    
    @staticmethod
    def calculate_variance(values: List[float]) -> float:
        """Calculate statistical variance in metrics"""
        return float(np.var(values)) if values else 0
    
    @staticmethod
    def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
        """Calculate confidence interval for a metric"""
        mean = np.mean(values)
        std_err = np.std(values, ddof=1) / np.sqrt(len(values))
        margin = 1.96 * std_err  # 95% CI
        
        return (mean - margin, mean + margin)