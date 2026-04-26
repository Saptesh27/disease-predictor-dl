"""Quick model sanity tests for sample symptom scenarios."""

from models.model_manager import ModelManager


def run():
    manager = ModelManager()
    manager.load_all()
    test_cases = [
        ("Respiratory", "fever sore throat runny nose body ache fatigue cough"),
        ("Diabetes", "frequent urination excessive thirst blurred vision fatigue weight loss"),
        ("Heart/Cardiac", "chest pain shortness of breath dizziness sweating left arm pain"),
        ("Gastric", "stomach pain nausea vomiting diarrhea loss of appetite bloating"),
        ("Neurological", "severe headache vision changes confusion memory loss dizziness"),
    ]
    for name, text in test_cases:
        print("=" * 70)
        print(name)
        print("Input:", text)
        result = manager.predict_both(text)
        print("BiLSTM top 3:")
        for r in result["bilstm"]["predictions"][:3]:
            print(f"- {r['disease']}: {r['percentage']}%")
        print("CNN top 3:")
        for r in result["cnn"]["predictions"][:3]:
            print(f"- {r['disease']}: {r['percentage']}%")
        print("Winner:", result["winner"])
        print("Agreement:", result["agreement"])


if __name__ == "__main__":
    run()
