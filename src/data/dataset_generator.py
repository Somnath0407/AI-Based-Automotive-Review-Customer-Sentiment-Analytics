"""
src/data/dataset_generator.py
───────────────────────────────
Synthetic ABSA dataset generator for automotive reviews.

Produces a balanced CSV with Review, Aspect, Sentiment columns
suitable for fine-tuning or quick demo analysis.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config_loader import get_aspects, get_sentiment_labels

# ── Template bank ─────────────────────────────────────────────────────────────

_POSITIVE_TEMPLATES: Dict[str, List[str]] = {
    "Vehicle": [
        "The vehicle has an outstanding build quality and eye-catching design.",
        "I love the exterior styling — this car turns heads everywhere.",
        "Excellent interior finish with premium materials throughout.",
        "The body design is modern and aerodynamic.",
        "Build quality is rock solid; no rattles or squeaks whatsoever.",
    ],
    "Engine": [
        "The engine delivers impressive horsepower with smooth acceleration.",
        "Turbo engine pulls effortlessly from low rpm — very responsive.",
        "The motor is refined and nearly silent at highway speeds.",
        "Excellent torque makes overtaking a breeze.",
        "The V6 engine is a real gem — powerful and efficient.",
    ],
    "Battery": [
        "Battery range exceeds 400 km on a full charge — outstanding.",
        "Charging from 20% to 80% takes only 35 minutes on a fast charger.",
        "The battery management system is intelligent and consistent.",
        "EV battery holds capacity well even in cold weather.",
        "Battery life is superb; no noticeable degradation after 50,000 km.",
    ],
    "Mileage": [
        "Fuel efficiency is remarkable — around 22 km/l in city conditions.",
        "The mileage on highways is exceptional, saving a lot on fuel.",
        "Great fuel economy for an SUV of this size.",
        "Consistent 18–20 kmpl even with AC running, very impressive.",
        "Petrol consumption is much lower than competitors in this segment.",
    ],
    "Safety": [
        "Five-star NCAP safety rating gives great peace of mind.",
        "Automatic emergency braking reacted perfectly in a test situation.",
        "Lane assist and blind-spot monitoring work flawlessly.",
        "Six airbags provide excellent protection for all occupants.",
        "The braking system is sharp and confidence-inspiring.",
    ],
    "Comfort": [
        "Extremely comfortable seats with excellent lumbar support.",
        "Ride quality is supple — potholes barely felt inside the cabin.",
        "Rear legroom is generous; great for long family journeys.",
        "Noise insulation is superb; cabin remains very quiet at speed.",
        "The ergonomic seat design reduces fatigue on long drives.",
    ],
    "Service": [
        "The dealership service team is professional and very responsive.",
        "Scheduled maintenance was completed faster than promised.",
        "Excellent after-sales support; all warranty claims resolved quickly.",
        "The service advisor explained every repair step clearly.",
        "Doorstep service pick-up is a brilliant convenience feature.",
    ],
    "Infotainment": [
        "The 12-inch touchscreen is crisp, responsive, and intuitive.",
        "Android Auto and Apple CarPlay work seamlessly every time.",
        "Navigation system reroutes quickly with up-to-date traffic data.",
        "Premium Bose audio system delivers concert-hall sound quality.",
        "Wireless connectivity is rock-solid; Bluetooth pairs instantly.",
    ],
    "Price": [
        "Excellent value for money given the features on offer.",
        "Competitively priced compared to rivals with fewer features.",
        "Affordable EMI options make this car accessible to many buyers.",
        "The price-to-performance ratio is outstanding.",
        "Considering the luxury feel, the cost is surprisingly reasonable.",
    ],
}

_NEGATIVE_TEMPLATES: Dict[str, List[str]] = {
    "Vehicle": [
        "The build quality feels cheap with visible panel gaps.",
        "Exterior styling looks dated compared to newer competitors.",
        "Interior plastics feel flimsy and scratch very easily.",
        "The car's body looks uninspiring and forgettable.",
        "Poor fit and finish — the dashboard trim rattles constantly.",
    ],
    "Engine": [
        "The engine feels underpowered for a vehicle of this size.",
        "Excessive engine noise at higher RPMs is very annoying.",
        "Slow acceleration makes merging onto highways a nerve-wracking experience.",
        "The motor vibrates noticeably at idle — unacceptable at this price.",
        "Turbo lag is significant and disrupts smooth driving.",
    ],
    "Battery": [
        "Battery range drops drastically in cold weather conditions.",
        "Charging time is far too long — over 1.5 hours even on a fast charger.",
        "Battery capacity has degraded noticeably within 18 months.",
        "The EV range of just 180 km is insufficient for daily commuting.",
        "Battery gets uncomfortably warm during rapid charging sessions.",
    ],
    "Mileage": [
        "Fuel economy is disappointingly low — barely 11 kmpl in city traffic.",
        "The claimed mileage is a far cry from the real-world figures.",
        "Gas consumption is too high for a small hatchback.",
        "Poor fuel efficiency for a car that is marketed as economical.",
        "Mileage drops steeply with AC and on rough roads.",
    ],
    "Safety": [
        "Lack of basic safety features like rear parking sensors is unacceptable.",
        "Only two airbags in a segment that now offers six as standard.",
        "Braking performance is mediocre — stopping distances are too long.",
        "No lane-departure warning or blind-spot monitoring at this price is shocking.",
        "The crash test rating is below average for this class.",
    ],
    "Comfort": [
        "The seats are hard and offer very little lumbar support.",
        "Rear passengers feel every bump — ride quality is unacceptably harsh.",
        "Cabin noise from wind and road is intrusive at highway speeds.",
        "Legroom in the second row is cramped even for average-height passengers.",
        "Vibrations through the steering wheel and floor are constant.",
    ],
    "Service": [
        "Dealership staff are rude and dismissive of customer concerns.",
        "Service waited three weeks with no update on a warranty repair.",
        "After-sales support is non-existent; calls go unanswered.",
        "Spare parts availability is terrible — had to wait two months.",
        "The service centre misdiagnosed the fault twice, wasting my time.",
    ],
    "Infotainment": [
        "The touchscreen lags badly and frequently freezes.",
        "Android Auto disconnects randomly every few minutes.",
        "Navigation maps are outdated and GPS signal is unreliable.",
        "Sound system produces tinny audio with no bass response.",
        "Bluetooth connection drops out constantly while playing music.",
    ],
    "Price": [
        "Overpriced for the features and build quality on offer.",
        "Expensive hidden charges at the dealership made it unaffordable.",
        "The cost of ownership is too high due to poor fuel efficiency.",
        "The price hike over the outgoing model is not justified.",
        "Too expensive for what is essentially a mid-spec family car.",
    ],
}

_NEUTRAL_TEMPLATES: Dict[str, List[str]] = {
    "Vehicle": [
        "The vehicle design is neither impressive nor outdated — just average.",
        "Build quality is acceptable but nothing remarkable.",
        "Exterior styling is clean but lacks a distinctive character.",
        "The body dimensions are practical for both city and highway use.",
        "Interior quality matches expectations for its price segment.",
    ],
    "Engine": [
        "The engine performance is adequate for typical daily commuting.",
        "Acceleration is acceptable, neither quick nor sluggish.",
        "Motor refinement is on par with others in this segment.",
        "Engine noise at high revs is noticeable but not alarming.",
        "Power output meets expectations for city driving.",
    ],
    "Battery": [
        "Battery range of 250 km is average for an EV in this price bracket.",
        "Charging time is comparable to most rivals.",
        "Battery capacity matches the manufacturer's published specifications.",
        "EV range is adequate for city commuting but tight for long trips.",
        "Battery performance in normal conditions is as expected.",
    ],
    "Mileage": [
        "Fuel economy is on par with others in the same segment.",
        "Mileage figures are neither outstanding nor disappointing.",
        "Fuel consumption is typical for a mid-size SUV.",
        "Real-world mileage matches the ARAI-certified figures closely.",
        "The fuel efficiency is acceptable given the engine displacement.",
    ],
    "Safety": [
        "Safety features meet regulatory standards but offer nothing extra.",
        "Four airbags are adequate but rivals now offer six as standard.",
        "ABS and EBD are present; advanced driver aids are an optional extra.",
        "Safety rating is average for this vehicle class.",
        "Braking performance is satisfactory in normal driving conditions.",
    ],
    "Comfort": [
        "Ride comfort is acceptable for city roads but firm on rough surfaces.",
        "Seats are reasonably comfortable for journeys up to two hours.",
        "Cabin noise levels are typical for a vehicle in this segment.",
        "Legroom in all rows is adequate for passengers of average height.",
        "Ride and handling balance is tuned toward practicality.",
    ],
    "Service": [
        "Dealership service is efficient but not particularly personalised.",
        "Service intervals and costs are similar to other brands.",
        "Warranty coverage is standard — nothing unusual to report.",
        "After-sales response time is average for the industry.",
        "Service centre staff are professional if somewhat impersonal.",
    ],
    "Infotainment": [
        "The infotainment system functions correctly but offers no standout features.",
        "Screen size and resolution are on par with competitors.",
        "Navigation system works adequately; no major complaints.",
        "Bluetooth connectivity is reliable and stable.",
        "Audio quality from the standard speakers is acceptable.",
    ],
    "Price": [
        "The price is neither a bargain nor overpriced for the segment.",
        "Cost of ownership is in line with similar vehicles.",
        "Pricing is competitive but not a decisive advantage.",
        "Value for money is reasonable given the feature set.",
        "EMI options are standard and not particularly attractive.",
    ],
}


class SyntheticDatasetGenerator:
    """
    Generates a synthetic, balanced ABSA dataset for automotive reviews.

    Each row represents one (Review, Aspect, Sentiment) annotation.
    Multiple rows can share the same review when multiple aspects are present.
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._aspects: List[str] = get_aspects()
        self._sentiments: List[str] = get_sentiment_labels()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, n_reviews: int = 200) -> pd.DataFrame:
        """
        Generate a DataFrame with approximately *n_reviews* unique reviews,
        each annotated for all detected aspects.

        Parameters
        ----------
        n_reviews : Approximate number of unique review strings to generate.

        Returns
        -------
        pd.DataFrame with columns: Review, Aspect, Sentiment
        """
        rows: List[Dict] = []
        sentiment_keys = ["Positive", "Negative", "Neutral"]

        for _ in range(n_reviews):
            # Pick 1–3 random aspects for a single review
            n_aspects = self._rng.randint(1, 3)
            chosen_aspects = self._rng.sample(self._aspects, k=min(n_aspects, len(self._aspects)))

            for aspect in chosen_aspects:
                sentiment = self._rng.choice(sentiment_keys)
                template_bank = {
                    "Positive": _POSITIVE_TEMPLATES,
                    "Negative": _NEGATIVE_TEMPLATES,
                    "Neutral": _NEUTRAL_TEMPLATES,
                }[sentiment]
                templates = template_bank.get(aspect, [f"The {aspect.lower()} is okay."])
                review = self._rng.choice(templates)
                rows.append({"Review": review, "Aspect": aspect, "Sentiment": sentiment})

        df = pd.DataFrame(rows)
        # Shuffle rows for a balanced dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

    def generate_balanced(self, samples_per_class: int = 30) -> pd.DataFrame:
        """
        Generate a strictly balanced dataset with exactly *samples_per_class*
        samples for each (Aspect × Sentiment) combination.
        """
        rows: List[Dict] = []
        for aspect in self._aspects:
            for sentiment in ["Positive", "Negative", "Neutral"]:
                template_bank = {
                    "Positive": _POSITIVE_TEMPLATES,
                    "Negative": _NEGATIVE_TEMPLATES,
                    "Neutral": _NEUTRAL_TEMPLATES,
                }[sentiment]
                templates = template_bank.get(aspect, [f"The {aspect.lower()} is okay."])
                for i in range(samples_per_class):
                    review = templates[i % len(templates)]
                    rows.append({"Review": review, "Aspect": aspect, "Sentiment": sentiment})

        df = pd.DataFrame(rows)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
