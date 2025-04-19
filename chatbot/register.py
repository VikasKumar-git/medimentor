import os
import json
import uuid

# Directory to store user data files
USERDATA_DIR = "userdata"
os.makedirs(USERDATA_DIR, exist_ok=True)

# Questions to ask
questions = {
    "gender": ["male", "female"],
    "age": "numerical",
    "Height": "numerical",
    "Weight": "numerical",
    "sysBP": "numerical",
    "diaBP": "numerical",
    "heartRate": "numerical",
    "cigsPerDay": "numerical",
    "smoking_status": ["never_smoked", "formerly_smoked", "smokes"],
    "chol_category": ["low_intake", "moderate_intake", "high_intake"],
    "glucose_category": ["low_intake", "moderate_intake", "high_intake", "very_high_intake"],
    "ever_married": ["yes", "no"],
    "Pregnancies": "numerical",
    "work_type": ["child", "never_worked", "self_employed", "govt_job", "private"],
    "Residence_type": ["urban", "rural"],
    "DiabetesPedigreeFunction": ["no_history", "low_history", "moderate_history", "high_history"],
    "exang": ["yes", "no"]
}

def ask_user_input():
    user_data = {}

    # Ask for name first
    name = input("Enter your full name: ").strip()
    user_data["name"] = name

    print("\nNow please enter your lifestyle details:\n")

    for field, options in questions.items():
        while True:
            if isinstance(options, list):
                print(f"{field} ({'/'.join(options)}): ", end="")
                answer = input().strip().lower()
                if answer in options:
                    user_data[field] = answer
                    break
                else:
                    print("❌ Invalid input. Try again.")
            else:
                try:
                    user_data[field] = float(input(f"{field} (numerical): ").strip())
                    break
                except ValueError:
                    print("❌ Please enter a valid number.")

    return user_data

def register_user():
    user_id = str(uuid.uuid4())[:8]  # Generate short unique user ID
    user_data = ask_user_input()
    user_data["user_id"] = user_id

    # Save to file
    filepath = os.path.join(USERDATA_DIR, f"{user_data["name"]}.json")
    with open(filepath, "w") as f:
        json.dump(user_data, f, indent=4)

    print(f"\n✅ Registration complete! Your User ID is: {user_id}")
    print(f"📁 Your data has been saved to: {filepath}")

if __name__ == "__main__":
    register_user()
