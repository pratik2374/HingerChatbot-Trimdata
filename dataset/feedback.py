import random
import pandas as pd

# Sample feedback pool categorized by product types
coffee_feedback = [
    "Coffee was too bitter.",
    "Loved the aroma of the coffee!",
    "Coffee served cold, not happy.",
    "Great taste, perfect start to the day.",
    "Could be a bit stronger.",
    "Espresso was excellent!",
    "Too much milk in the cappuccino.",
    "Coffee felt watered down.",
    "Perfectly brewed, will order again.",
    "Good value for money coffee."
]

chai_feedback = [
    "Chai was too sweet.",
    "Loved the masala chai!",
    "Chai had a strange aftertaste.",
    "Hot and refreshing.",
    "Chai quantity was less than expected.",
    "Best chai I’ve had in a while.",
    "Chai was cold and disappointing.",
    "Fragrant and flavorful!",
    "Tasted homemade, loved it.",
    "Needs more ginger in chai."
]

sandwich_feedback = [
    "Bread was stale.",
    "Delicious sandwich!",
    "Too much mayo in the sandwich.",
    "Good mix of veggies and cheese.",
    "Sandwich was a bit dry.",
    "Loved the grilled paneer sandwich.",
    "Could use better quality bread.",
    "Great portion size for the price.",
    "Fresh ingredients, nice taste.",
    "Sandwich was soggy."
]

snacks_feedback = [
    "Samosa was crispy and tasty.",
    "Not enough filling in the wrap.",
    "Tikki was undercooked.",
    "Loved the chutneys with the snacks.",
    "Very oily snack items.",
    "Perfect evening bite.",
    "Packaging could be better.",
    "Snacks were fresh and warm.",
    "Good portion size, fair price.",
    "Taste was okay, nothing special."
]

meals_feedback = [
    "Rice was dry and undercooked.",
    "Nice thali, felt homely.",
    "Curry was too spicy.",
    "Loved the dal makhani.",
    "Packaging leaked during delivery.",
    "Healthy and tasty meal.",
    "Could use more variety in the meals.",
    "Great portion and taste.",
    "Meal wasn’t hot on arrival.",
    "Affordable and filling meal combo."
]

# Randomly select feedback from all categories
all_feedback = coffee_feedback + chai_feedback + sandwich_feedback + snacks_feedback + meals_feedback
feedback_column = random.choices(all_feedback, k=50)

# Create a DataFrame to display
df = pd.DataFrame({"Feedback": feedback_column})
original_df = pd.read_csv("final_merged_output_sample_testing (1) - Copy of final_merged_output_sample_testing (1).csv", low_memory=False).head(50)
merged_df = pd.concat([original_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

merged_df.to_csv("feedback_output.csv", index=False)
