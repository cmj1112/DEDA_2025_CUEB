import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import textwrap
import random
import platform
import os

# Set global settings
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # Standard fonts
plt.rcParams['axes.unicode_minus'] = False

# ===================== Create Beer Dataset =====================
# Generate simulated data for 200 beers
np.random.seed(42)

# Beer types
beer_types = ['Pilsner', 'IPA', 'Stout', 'Wheat Beer', 'Amber Ale', 'Sour Beer', 'Lager', 'Belgian Style']
n_samples = 200

data = {
    'Beer Name': [f'Beer #{i + 1}' for i in range(n_samples)],
    'Beer Type': np.random.choice(beer_types, n_samples, p=[0.15, 0.15, 0.1, 0.15, 0.1, 0.05, 0.2, 0.1]),
    'Flavor Profile': np.random.choice(['Light', 'Medium', 'Rich'], n_samples, p=[0.4, 0.4, 0.2]),
    'Bitterness': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.5, 0.3, 0.2]),
    'Aroma': np.random.choice(['Floral', 'Fruity', 'Malty', 'Roasty'], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
    'ABV': np.random.choice(['Low <4%', 'Medium 4-6%', 'High >6%'], n_samples, p=[0.4, 0.4, 0.2]),
    'Occasion': np.random.choice(['Casual', 'Social', 'Food Pairing', 'Celebration'], n_samples),
    'Season': np.random.choice(['Summer', 'Winter', 'All Seasons'], n_samples, p=[0.4, 0.3, 0.3]),
    'Rating': np.round(np.random.uniform(3.5, 5.0, n_samples), 1)
}

df = pd.DataFrame(data)

# Add type encoding for model training
type_mapping = {beer: idx for idx, beer in enumerate(beer_types)}
df['Type Code'] = df['Beer Type'].map(type_mapping)

# Feature encoding
feature_mappings = {
    'Flavor Profile': {'Light': 0, 'Medium': 1, 'Rich': 2},
    'Bitterness': {'Low': 0, 'Medium': 1, 'High': 2},
    'Aroma': {'Floral': 0, 'Fruity': 1, 'Malty': 2, 'Roasty': 3},
    'ABV': {'Low <4%': 0, 'Medium 4-6%': 1, 'High >6%': 2},
    'Occasion': {'Casual': 0, 'Social': 1, 'Food Pairing': 2, 'Celebration': 3},
    'Season': {'Summer': 0, 'Winter': 1, 'All Seasons': 2}
}

for feature, mapping in feature_mappings.items():
    df[f'{feature} Code'] = df[feature].map(mapping)

# Feature columns
features = ['Flavor Profile Code', 'Bitterness Code', 'Aroma Code', 'ABV Code', 'Occasion Code', 'Season Code']
X = df[features]
y = df['Type Code']

# ===================== Train Decision Tree Model =====================
# Create decision tree classifier
clf = DecisionTreeClassifier(
    max_depth=7,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Train model
clf.fit(X, y)

# ===================== Visualize Decision Tree =====================
plt.figure(figsize=(24, 16))

# Custom feature names (user-friendly questions)
custom_feature_names = [
    "Preferred flavor profile?",
    "Bitterness tolerance?",
    "Aroma preference?",
    "Desired alcohol level?",
    "Drinking occasion?",
    "Preferred season?"
]

# Custom class names
custom_class_names = beer_types

# Plot decision tree
plot_tree(clf,
          feature_names=custom_feature_names,
          class_names=custom_class_names,
          filled=True,
          rounded=True,
          fontsize=10,
          proportion=True,
          impurity=False)

plt.title("Beer Recommendation Decision Tree", fontsize=20, pad=20)
plt.tight_layout()
plt.savefig('beer_decision_tree.png', dpi=120, bbox_inches='tight')
plt.close()  # Close plot to avoid display

# ===================== Beer Knowledge Base =====================
beer_knowledge = {
    'Pilsner': {
        'desc': "Crisp golden beer with gentle malt sweetness and balanced bitterness",
        'fun_fact': "Pilsner was first brewed in 1842 in Plzeň, Czech Republic",
        'food_pair': "Perfect with grilled chicken, salads, or seafood"
    },
    'IPA': {
        'desc': "Features bold hop aromas and bitterness with citrus or pine notes",
        'fun_fact': "IPA was originally brewed with extra hops to survive long sea voyages to India",
        'food_pair': "Pairs well with spicy foods, burgers, and barbecue"
    },
    'Stout': {
        'desc': "Dark beer with roasted coffee, chocolate, or caramel flavors",
        'fun_fact': "Guinness popularized the stout style in the 18th century",
        'food_pair': "Excellent with chocolate desserts, oysters, or smoked meats"
    },
    'Wheat Beer': {
        'desc': "Cloudy pale beer with banana and clove aromas, refreshing taste",
        'fun_fact': "German law requires wheat beers to contain at least 50% wheat malt",
        'food_pair': "Best with white meats, salads, or fruit desserts"
    },
    'Amber Ale': {
        'desc': "Balanced beer with malt sweetness and hop bitterness, caramel notes",
        'fun_fact': "Amber Ale was a pioneer style in the American craft beer revolution",
        'food_pair': "Great with pizza, grilled meats, or cheese platters"
    },
    'Sour Beer': {
        'desc': "Tart beer often with fruity notes, refreshing and complex",
        'fun_fact': "Sour beer is one of the oldest beer styles, using wild yeast fermentation",
        'food_pair': "Creates wonderful contrasts with seafood, salads, or creamy desserts"
    },
    'Lager': {
        'desc': "Clean and crisp beer with prominent malt character, low bitterness",
        'fun_fact': "Lager means 'to store' in German, referring to its cold fermentation process",
        'food_pair': "Versatile pairing with all foods, especially fast food"
    },
    'Belgian Style': {
        'desc': "Diverse beers with spicy or fruity notes, complex and intriguing",
        'fun_fact': "Belgian Trappist beers have been brewed by monks for centuries",
        'food_pair': "Complements stews, shellfish, or cream-based pasta dishes"
    }
}


# ===================== Interactive Beer Recommendation System =====================
def beer_recommendation_system():
    print("""
  🍻 Welcome to the Beer Tasting Master System! 🍻

  We'll help you find your perfect beer through a few simple questions
  Please select the option that best matches your preference for each question
  """)

    # Collect user preferences
    answers = {}

    print("\n===== Flavor Preference =====")
    print("1. Light - Refreshing and easy-drinking")
    print("2. Medium - Balanced flavor for various occasions")
    print("3. Rich - Complex and full-bodied")
    answers['Flavor'] = int(input("Choose (1-3): ")) - 1

    print("\n===== Bitterness Tolerance =====")
    print("1. Low - Prefer mild bitterness, sweet flavors")
    print("2. Medium - Enjoy balanced bitterness")
    print("3. High - Appreciate bold hop bitterness")
    answers['Bitterness'] = int(input("Choose (1-3): ")) - 1

    print("\n===== Aroma Preference =====")
    print("1. Floral - Hoppy, floral scents")
    print("2. Fruity - Citrus, tropical fruit notes")
    print("3. Malty - Bread, grain aromas")
    print("4. Roasty - Coffee, chocolate, roasted notes")
    answers['Aroma'] = int(input("Choose (1-4): ")) - 1

    print("\n===== Alcohol Preference =====")
    print("1. Low <4% - Easy-drinking session beers")
    print("2. Medium 4-6% - Balanced alcohol presence")
    print("3. High >6% - Strong, warming beers")
    answers['ABV'] = int(input("Choose (1-3): ")) - 1

    print("\n===== Drinking Occasion =====")
    print("1. Casual - Relaxing after work")
    print("2. Social - Sharing with friends")
    print("3. Food Pairing - Enhancing meals")
    print("4. Celebration - Special occasions")
    answers['Occasion'] = int(input("Choose (1-4): ")) - 1

    print("\n===== Season Preference =====")
    print("1. Summer - Thirst-quenching for hot weather")
    print("2. Winter - Warming for cold seasons")
    print("3. All Seasons - Year-round enjoyment")
    answers['Season'] = int(input("Choose (1-3): ")) - 1

    # Convert to model input
    input_data = [
        answers['Flavor'],
        answers['Bitterness'],
        answers['Aroma'],
        answers['ABV'],
        answers['Occasion'],
        answers['Season']
    ]

    # Predict beer type
    prediction = clf.predict([input_data])[0]
    beer_type = beer_types[prediction]
    proba = clf.predict_proba([input_data])[0][prediction]

    # Select a specific beer from dataset
    beer_options = df[df['Beer Type'] == beer_type]
    if not beer_options.empty:
        recommended_beer = beer_options.sample(1).iloc[0]
    else:
        # Default recommendation if no match
        recommended_beer = {'Beer Name': f'Classic {beer_type}', 'Rating': 4.5}

    # Get beer knowledge
    knowledge = beer_knowledge.get(beer_type, {})

    # Print recommendation
    print("\n" + "=" * 60)
    print(f"🌟 YOUR PERFECT BEER: {beer_type} 🌟")
    print(f"🍺 RECOMMENDED BREW: {recommended_beer['Beer Name']} (Rating: {recommended_beer['Rating']}/5.0)")
    print(f"📊 MATCH SCORE: {proba * 100:.1f}%")
    print("=" * 60)
    print(f"💡 CHARACTERISTICS: {knowledge.get('desc', '')}")
    print(f"🎓 FUN FACT: {knowledge.get('fun_fact', '')}")
    print(f"🍽️ FOOD PAIRING: {knowledge.get('food_pair', '')}")
    print("=" * 60)

    # Generate beer passport
    generate_beer_passport(beer_type, recommended_beer['Beer Name'], proba)

    print("\nYour 'Beer Passport' has been saved as beer_passport.png!")
    print("Share it with friends and explore the world of beer! 🍻")


def generate_beer_passport(beer_type, beer_name, match_percent):
    # Create passport image
    img = Image.new('RGB', (800, 500), color=(240, 240, 220))  # Cream background
    draw = ImageDraw.Draw(img)

    # Font handling
    try:
        # Try to use standard fonts
        title_font = ImageFont.truetype("arial.ttf", 36)
        type_font = ImageFont.truetype("arial.ttf", 32)
        beer_font = ImageFont.truetype("arial.ttf", 28)
        match_font = ImageFont.truetype("arial.ttf", 24)
        fact_font = ImageFont.truetype("arial.ttf", 18)
    except:
        # Fallback to default fonts
        title_font = ImageFont.load_default()
        type_font = ImageFont.load_default()
        beer_font = ImageFont.load_default()
        match_font = ImageFont.load_default()
        fact_font = ImageFont.load_default()

    # Add title
    draw.text((400, 40), "BEER PASSPORT", fill=(0, 0, 0), font=title_font, anchor="mm")

    # Add divider
    draw.line([(50, 80), (750, 80)], fill=(150, 75, 0), width=2)

    # Add beer type
    draw.text((400, 130), f"RECOMMENDED STYLE: {beer_type}", fill=(200, 0, 0), font=type_font, anchor="mm")

    # Add specific beer
    draw.text((400, 180), f"RECOMMENDED BEER: {beer_name}", fill=(0, 0, 0), font=beer_font, anchor="mm")

    # Add match score
    draw.text((400, 230), f"MATCH SCORE: {match_percent * 100:.1f}%", fill=(0, 100, 0), font=match_font, anchor="mm")

    # Add beer knowledge
    knowledge = beer_knowledge.get(beer_type, {})

    # Characteristics
    desc = knowledge.get('desc', '')
    desc_y = 280
    draw.text((100, desc_y), f"CHARACTERISTICS: {desc}", fill=(0, 0, 0), font=fact_font)

    # Fun fact
    fun_fact = knowledge.get('fun_fact', '')
    fact_y = desc_y + 40
    draw.text((100, fact_y), f"FUN FACT: {fun_fact}", fill=(0, 0, 0), font=fact_font)

    # Food pairing
    food_pair = knowledge.get('food_pair', '')
    food_y = fact_y + 40
    draw.text((100, food_y), f"FOOD PAIRING: {food_pair}", fill=(0, 0, 0), font=fact_font)

    # Add certification stamp
    draw.ellipse([(650, 350), (750, 450)], outline=(200, 0, 0), width=3)
    draw.text((700, 400), "CERTIFIED", fill=(200, 0, 0), font=fact_font, anchor="mm")

    # Save passport
    img.save('beer_passport.png')


# ===================== Execute System =====================
if __name__ == "__main__":
    # Start recommendation system
    beer_recommendation_system()

    print("\nThank you for using the Beer Tasting Master System! Cheers! 🍻")