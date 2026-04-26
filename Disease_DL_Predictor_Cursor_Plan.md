# 🏥 Disease Predictor — Deep Learning Upgrade
### BiLSTM vs CNN Text Classifier + Streamlit UI + Kaggle Dataset

> **Stack**: Python · TensorFlow/Keras · BiLSTM · CNN · spaCy · Streamlit · Plotly  
> **Dataset**: Kaggle "Disease Symptom Prediction" (132 diseases, 4920 rows)  
> **No GPU Required** · 100% Free · Local Training  
> **Time Estimate**: 4–6 hours full build

---

## 🗂️ FIRST — Project Structure

### 👉 Cursor Prompt #1 — Scaffold Full Project

```
Create a Python project with this exact folder structure:

disease_dl_predictor/
├── app.py                          # Main Streamlit UI
├── train.py                        # Script to train both models
├── nlp/
│   ├── __init__.py
│   ├── preprocessor.py             # Text cleaning + tokenization
│   ├── symptom_extractor.py        # spaCy PhraseMatcher for symptoms
│   └── drug_extractor.py           # spaCy PhraseMatcher for drug names
├── models/
│   ├── __init__.py
│   ├── bilstm_model.py             # BiLSTM model architecture
│   ├── cnn_model.py                # CNN text classifier architecture
│   └── model_manager.py            # Load, save, predict with both models
├── training/
│   ├── __init__.py
│   ├── data_loader.py              # Load + prepare Kaggle dataset
│   ├── trainer.py                  # Train both models, save weights
│   └── evaluator.py                # Evaluate, compare, generate metrics
├── visualization/
│   ├── __init__.py
│   └── charts.py                   # All Plotly chart functions
├── saved_models/
│   ├── bilstm/                     # Saved BiLSTM weights
│   └── cnn/                        # Saved CNN weights
├── data/
│   ├── dataset.csv                 # Kaggle disease-symptom dataset
│   ├── tokenizer.pkl               # Saved Keras tokenizer
│   ├── label_encoder.pkl           # Saved label encoder
│   └── drugs.csv                   # Drug → Disease mapping (from before)
├── config/
│   └── settings.py                 # All hyperparameters + paths
├── requirements.txt
├── setup.py
└── README.md

Create every file with a comment header. Add pass statements as placeholders.
```

---

## ⚙️ STEP 1 — Requirements & Settings

### 👉 Cursor Prompt #2 — requirements.txt + settings.py

```
Create requirements.txt with these packages:

streamlit>=1.32.0
tensorflow>=2.13.0
keras>=2.13.0
scikit-learn>=1.3.0
spacy>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
nltk>=3.8.0
fuzzywuzzy>=0.18.0
python-Levenshtein>=0.21.0
pickle5>=0.0.12
requests>=2.31.0
kaggle>=1.5.16

Then create config/settings.py with a Settings class containing:

# Dataset
DATASET_PATH = "data/dataset.csv"
DRUGS_CSV_PATH = "data/drugs.csv"
TOKENIZER_PATH = "data/tokenizer.pkl"
LABEL_ENCODER_PATH = "data/label_encoder.pkl"

# Text Processing
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 5000
EMBEDDING_DIM = 128

# BiLSTM Hyperparameters
BILSTM_UNITS = 64
BILSTM_DROPOUT = 0.3
BILSTM_RECURRENT_DROPOUT = 0.3

# CNN Hyperparameters
CNN_FILTERS = 128
CNN_KERNEL_SIZES = [2, 3, 4]
CNN_DROPOUT = 0.5

# Training
EPOCHS = 15
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Model Save Paths
BILSTM_MODEL_PATH = "saved_models/bilstm/model.h5"
CNN_MODEL_PATH = "saved_models/cnn/model.h5"

# App
TOP_N_DISEASES = 5
CONFIDENCE_THRESHOLD = 0.1

Export a single settings instance.
```

---

## 📦 STEP 2 — Download Kaggle Dataset

### 👉 Cursor Prompt #3 — data_loader.py + setup.py

```
In training/data_loader.py, write a DataLoader class:

1. The Kaggle dataset being used is:
   Name: "Disease Symptom Prediction"
   URL: https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
   Files needed: dataset.csv
   
   dataset.csv structure:
   - Column "Disease": disease name string
   - Columns "Symptom_1" through "Symptom_17": symptom name or NaN
   - Each row is one disease-symptom combination
   - 132 unique diseases, 4920 rows total

2. Method load_raw_data(self) -> pd.DataFrame:
   - Load dataset.csv using pandas
   - Print shape and first 5 rows
   - Return dataframe

3. Method prepare_features(self, df: pd.DataFrame) -> tuple:
   
   EXPLAIN IN COMMENT: The dataset has symptoms spread across 17 columns
   (Symptom_1, Symptom_2... Symptom_17). We need to combine them into
   one text string per row for our text classifier.
   
   Steps:
   a) Get all symptom columns: cols = [c for c in df.columns if "Symptom" in c]
   b) For each row, join non-null symptom values into one string:
      "itching skin rash nodal skin eruptions"
   c) Clean each combined text: lowercase, strip extra spaces
   d) X = list of combined symptom text strings
   e) y = list of disease labels (df["Disease"])
   f) Return X, y

4. Method encode_labels(self, y: list) -> tuple:
   - Use sklearn LabelEncoder
   - Fit and transform y
   - Save encoder to LABEL_ENCODER_PATH using pickle
   - Return (encoded_y, encoder, num_classes)
   
   EXPLAIN: LabelEncoder converts disease strings to integers.
   "Diabetes" → 23, "Flu" → 41, etc.
   We save it so we can reverse the mapping during prediction.

5. Method tokenize_text(self, X: list) -> tuple:
   
   EXPLAIN IN COMMENT: Keras Tokenizer converts words to integers.
   "fever headache cough" → [34, 12, 8]
   Then we pad sequences to same length so the model gets 
   uniform input shape.
   
   Steps:
   a) Create Keras Tokenizer with num_words=MAX_VOCAB_SIZE
   b) Fit on X texts: tokenizer.fit_on_texts(X)
   c) Convert to sequences: sequences = tokenizer.texts_to_sequences(X)
   d) Pad sequences to MAX_SEQUENCE_LENGTH using pad_sequences
   e) Save tokenizer to TOKENIZER_PATH using pickle
   f) Return (padded_sequences, tokenizer)

6. Method train_test_split_data(self, X_pad, y_enc) -> tuple:
   - Use sklearn train_test_split
   - Split: 80% train, 10% val, 10% test
   - Set random_state=RANDOM_SEED
   - Return X_train, X_val, X_test, y_train, y_val, y_test

7. Method prepare_all(self) -> dict:
   - Call all above methods in order
   - Return dict with all prepared data + tokenizer + encoder
   - Print summary: num_classes, vocab_size, train_size, test_size

Also create setup.py (plain Python script) that:
1. Runs: python -m spacy download en_core_web_sm
2. Creates all required folders: saved_models/bilstm, saved_models/cnn, data/
3. Checks if data/dataset.csv exists — if not, prints download instructions:
   "Please download dataset.csv from:
    https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
    Place it in the data/ folder"
4. Prints setup complete message
```

---

## 🧠 STEP 3 — BiLSTM Model

### 👉 Cursor Prompt #4 — bilstm_model.py

```
In models/bilstm_model.py, build the BiLSTM model using Keras.

Write a BiLSTMModel class:

1. __init__(self, vocab_size, num_classes, embedding_dim, max_length):
   - Store all params as instance variables
   - self.model = None
   - self.history = None

2. build(self) -> keras.Model:
   
   EXPLAIN EVERY LAYER IN DETAIL WITH COMMENTS:
   
   Use Keras Sequential or Functional API:
   
   Layer 1 - Embedding:
   Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length)
   
   EXPLAIN: Embedding layer converts integer word IDs into dense vectors.
   Each word gets a unique vector of size embedding_dim (128).
   Example: word "fever" (id=34) → [0.23, -0.12, 0.45, ...] (128 numbers)
   The model LEARNS these vectors during training.
   
   Layer 2 - Spatial Dropout:
   SpatialDropout1D(0.2)
   
   EXPLAIN: Drops entire feature maps randomly during training.
   Prevents overfitting better than regular dropout for embeddings.
   
   Layer 3 - Bidirectional LSTM:
   Bidirectional(LSTM(units=BILSTM_UNITS, 
                      dropout=BILSTM_DROPOUT,
                      recurrent_dropout=BILSTM_RECURRENT_DROPOUT,
                      return_sequences=True))
   
   EXPLAIN IN DETAIL: 
   - LSTM reads sequence step by step, maintaining hidden state (memory)
   - Bidirectional wraps LSTM to read sequence BOTH forward AND backward
   - Forward LSTM: "fever" → "headache" → "cough" (left to right)
   - Backward LSTM: "cough" → "headache" → "fever" (right to left)
   - Both outputs concatenated → richer representation of each word
   - return_sequences=True → output at every timestep (needed for next LSTM)
   
   Layer 4 - Second Bidirectional LSTM:
   Bidirectional(LSTM(units=BILSTM_UNITS//2,
                      dropout=BILSTM_DROPOUT))
   
   EXPLAIN: Second LSTM layer learns higher-level patterns from first layer.
   Units halved to create a funnel shape — compress information.
   return_sequences=False → only outputs final state (summary of whole sequence)
   
   Layer 5 - Dense + BatchNorm + Dropout:
   Dense(128, activation="relu")
   BatchNormalization()
   Dropout(0.4)
   
   EXPLAIN: Dense layer learns combinations of LSTM features.
   BatchNormalization normalizes activations — stabilizes training.
   Dropout randomly zeros 40% of neurons — prevents overfitting.
   
   Layer 6 - Output Dense:
   Dense(num_classes, activation="softmax")
   
   EXPLAIN: Final layer has one neuron per disease class.
   Softmax converts raw scores to probabilities that sum to 1.0.
   Example output: [0.02, 0.87, 0.01, ...] → 87% chance of disease class 1.
   
   Compile with:
   optimizer = Adam(learning_rate=LEARNING_RATE)
   loss = "sparse_categorical_crossentropy"
   metrics = ["accuracy"]
   
   self.model = model
   return model

3. get_callbacks(self) -> list:
   Return list of Keras callbacks:
   
   a) EarlyStopping:
      monitor="val_loss", patience=5, restore_best_weights=True
      EXPLAIN: Stops training if val_loss doesn't improve for 5 epochs.
      Saves time and prevents overfitting.
   
   b) ReduceLROnPlateau:
      monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001
      EXPLAIN: Reduces learning rate when training plateaus.
      Helps model fine-tune as it gets closer to optimal.
   
   c) ModelCheckpoint:
      filepath=BILSTM_MODEL_PATH, save_best_only=True
      EXPLAIN: Automatically saves best model weights during training.

4. train(self, X_train, y_train, X_val, y_val) -> dict:
   - Build model if not built
   - Fit with callbacks
   - Store history
   - Return history.history dict (contains loss/accuracy per epoch)

5. predict(self, X_input) -> np.ndarray:
   - Run model.predict()
   - Return probability array shape (1, num_classes)

6. get_model_summary(self) -> str:
   - Capture model.summary() output as string
   - Return it

7. save(self) and load(self) methods using model.save() and load_model()
```

---

## 🔲 STEP 4 — CNN Text Classifier Model

### 👉 Cursor Prompt #5 — cnn_model.py

```
In models/cnn_model.py, build a CNN Text Classifier using Keras Functional API.
This model uses multiple parallel convolution filters of different sizes.

Write a CNNModel class:

1. __init__(self, vocab_size, num_classes, embedding_dim, max_length):
   - Store all params
   - self.model = None
   - self.history = None

2. build(self) -> keras.Model:
   
   EXPLAIN THE FULL CNN ARCHITECTURE WITH DETAILED COMMENTS:
   
   Use Keras Functional API (NOT Sequential) because we need parallel branches:
   
   Input layer:
   inputs = Input(shape=(max_length,))
   
   EXPLAIN: Input shape is (max_length,) — a sequence of max_length integers.
   
   Shared Embedding layer:
   x = Embedding(vocab_size, embedding_dim)(inputs)
   x = SpatialDropout1D(0.2)(x)
   
   EXPLAIN: Same embedding as BiLSTM — converts word IDs to dense vectors.
   Output shape: (batch_size, max_length, embedding_dim)
   
   Parallel Convolution Branches (THIS IS THE KEY CNN PART):
   Create 3 separate Conv1D branches with kernel sizes [2, 3, 4]:
   
   For each kernel_size in [2, 3, 4]:
       branch = Conv1D(filters=CNN_FILTERS, 
                       kernel_size=kernel_size,
                       activation="relu",
                       padding="same")(x)
       
       EXPLAIN Conv1D IN DETAIL:
       - Conv1D slides a window of size kernel_size across the text sequence
       - kernel_size=2: looks at 2-word combinations (bigrams)
         "high fever" → one feature, "body pain" → another feature
       - kernel_size=3: looks at 3-word combinations (trigrams)
         "severe chest pain" → one feature
       - kernel_size=4: looks at 4-word phrases
       - filters=128 means 128 different pattern detectors per kernel
       - activation="relu" → keeps only positive activations
       
       branch = GlobalMaxPooling1D()(branch)
       
       EXPLAIN GlobalMaxPooling1D:
       - Takes the maximum value across all positions for each filter
       - Captures the STRONGEST signal each filter detected anywhere in text
       - Reduces output from (max_length, 128) to (128,) — one value per filter
       - This makes the model position-invariant: "fever" detected regardless
         of whether it's at start or end of symptom text
   
   Concatenate all branches:
   merged = Concatenate()([branch2, branch3, branch4])
   
   EXPLAIN: Concatenate combines outputs from all 3 kernel sizes.
   Final shape: (128 * 3,) = (384,)
   Now the model has features from bigrams, trigrams AND 4-grams combined.
   
   Dense layers:
   x = Dense(256, activation="relu")(merged)
   x = BatchNormalization()(x)
   x = Dropout(CNN_DROPOUT)(x)
   x = Dense(128, activation="relu")(x)
   x = Dropout(0.3)(x)
   outputs = Dense(num_classes, activation="softmax")(x)
   
   Build model:
   model = Model(inputs=inputs, outputs=outputs)
   
   Compile with Adam optimizer, sparse_categorical_crossentropy loss, accuracy metric.
   
   self.model = model
   return model

3. get_callbacks(self) -> list:
   Same callbacks as BiLSTM but save to CNN_MODEL_PATH

4. train(self, X_train, y_train, X_val, y_val) -> dict:
   Same as BiLSTM

5. predict(self, X_input) -> np.ndarray:
   Same as BiLSTM

6. save(self) and load(self) methods
```

---

## 🏋️ STEP 5 — Trainer & Evaluator

### 👉 Cursor Prompt #6 — trainer.py + evaluator.py

```
In training/trainer.py, write a Trainer class:

1. __init__(self):
   - Initialize DataLoader
   - self.data = None
   - self.bilstm = None
   - self.cnn = None

2. prepare_data(self):
   - Call DataLoader.prepare_all()
   - Store in self.data

3. build_models(self):
   - Build BiLSTMModel with correct vocab_size, num_classes
   - Build CNNModel with same params
   - Store in self.bilstm, self.cnn
   - Print both model summaries

4. train_bilstm(self) -> dict:
   - Train BiLSTM on training data
   - Print training progress
   - Save model
   - Return history dict

5. train_cnn(self) -> dict:
   - Train CNN on training data
   - Save model
   - Return history dict

6. train_both(self) -> dict:
   - Call prepare_data()
   - Call build_models()
   - Print "Training BiLSTM..."
   - Train BiLSTM, store history
   - Print "Training CNN..."
   - Train CNN, store history
   - Call evaluator to compare both
   - Return combined results dict

---

In training/evaluator.py, write an Evaluator class:

1. __init__(self, bilstm_model, cnn_model, data):
   - Store model references and test data

2. evaluate_model(self, model, X_test, y_test) -> dict:
   - Run model.predict() on test set
   - Calculate metrics:
     * accuracy = accuracy_score(y_test, y_pred)
     * precision = precision_score(y_test, y_pred, average="weighted")
     * recall = recall_score(y_test, y_pred, average="weighted")
     * f1 = f1_score(y_test, y_pred, average="weighted")
   - Return dict with all metrics

3. compare_both(self, X_test, y_test) -> dict:
   - Evaluate BiLSTM → get metrics dict
   - Evaluate CNN → get metrics dict
   - Return comparison dict:
     {
       "bilstm": {accuracy, precision, recall, f1, training_time},
       "cnn": {accuracy, precision, recall, f1, training_time},
       "winner": "BiLSTM" or "CNN" (whichever has higher F1),
       "bilstm_history": {...epoch data...},
       "cnn_history": {...epoch data...}
     }

4. get_per_class_report(self, model, X_test, y_test, label_encoder) -> pd.DataFrame:
   - Run classification_report(output_dict=True)
   - Convert to DataFrame
   - Return sorted by f1-score descending
```

---

## 🤒 STEP 6 — spaCy Extractors (Updated for DL)

### 👉 Cursor Prompt #7 — preprocessor.py + symptom_extractor.py

```
In nlp/preprocessor.py, write a TextPreprocessor class:

1. __init__(self, tokenizer, max_length):
   - Store tokenizer (loaded Keras tokenizer) and max_length

2. clean_text(self, text: str) -> str:
   - Lowercase
   - Remove special characters except letters and spaces
   - Remove extra whitespace
   - Return cleaned string

3. remove_dosage(self, text: str) -> str:
   - Use regex to remove: mg, ml, mcg, g, iu, units, numbers
   - Remove frequency words: "twice daily", "TDS", "BD", "OD", "QID"
   - Return cleaned text

4. text_to_sequence(self, text: str) -> np.ndarray:
   - Clean the text
   - Convert to sequence: tokenizer.texts_to_sequences([text])
   - Pad to max_length: pad_sequences(seq, maxlen=max_length)
   - Return padded numpy array ready for model input
   
   EXPLAIN: This converts raw input text into the exact format
   the BiLSTM and CNN models expect. Without this step,
   models cannot process user input.

---

In nlp/symptom_extractor.py, write a SymptomExtractor class using spaCy PhraseMatcher:

1. __init__(self, symptoms_list: list):
   - Load spaCy: self.nlp = spacy.load("en_core_web_sm")
   - Create PhraseMatcher with attr="LOWER"
   - Add all symptoms from symptoms_list as patterns
   - Create negation PhraseMatcher for: ["no", "not", "without", 
     "denies", "absence of", "no history of"]

2. extract_symptoms(self, text: str) -> list[str]:
   - Process text with spaCy
   - Run PhraseMatcher
   - For each match: check if negated (look 3 tokens before)
   - Return list of non-negated symptom strings

3. symptoms_to_text(self, symptoms: list[str]) -> str:
   - Join symptom list into a single string
   - This string gets fed into the DL models
   - Return joined string
   
   EXPLAIN: The DL models take raw text as input. By converting
   extracted symptoms back to text, we feed clean symptom-only
   text to both models — removing noise from patient description.
```

---

## 🔮 STEP 7 — Model Manager (Unified Prediction)

### 👉 Cursor Prompt #8 — model_manager.py

```
In models/model_manager.py, write a ModelManager class that handles
loading both models and running predictions.

1. __init__(self):
   - self.bilstm = None
   - self.cnn = None
   - self.tokenizer = None
   - self.label_encoder = None
   - self.preprocessor = None
   - self.symptom_extractor = None
   - self.is_loaded = False

2. load_all(self):
   - Load tokenizer from pickle file
   - Load label encoder from pickle file
   - Initialize TextPreprocessor
   - Load BiLSTM model weights
   - Load CNN model weights
   - Get symptom list from label encoder classes
   - Initialize SymptomExtractor with dataset symptom vocabulary
   - Set self.is_loaded = True
   - Print "✅ Both models loaded successfully"
   - Handle FileNotFoundError — tell user to run train.py first

3. predict_both(self, input_text: str, input_mode: str = "symptoms") -> dict:
   
   EXPLAIN: This is the core prediction method called by the UI.
   input_mode = "symptoms" → raw symptom description
   input_mode = "prescription" → drug names from prescription
   
   Steps:
   a) If input_mode == "symptoms":
      - Extract symptoms using SymptomExtractor
      - Convert symptoms list to clean text string
   
   b) If input_mode == "prescription":
      - Use DrugExtractor to get drug names
      - Map drug names to symptom-like descriptions
      - Create text from drug-disease mapping
   
   c) Convert text to model input sequence:
      X_input = preprocessor.text_to_sequence(clean_text)
   
   d) Get BiLSTM predictions:
      bilstm_probs = bilstm.predict(X_input)[0]
      EXPLAIN: predict() returns array of shape (1, num_classes).
      [0] gets the first (and only) sample's probabilities.
   
   e) Get CNN predictions:
      cnn_probs = cnn.predict(X_input)[0]
   
   f) Get top N diseases for each model:
      
      For BiLSTM:
      top_indices = np.argsort(bilstm_probs)[::-1][:TOP_N_DISEASES]
      bilstm_results = [
          {
            "disease": label_encoder.inverse_transform([i])[0],
            "probability": float(bilstm_probs[i]),
            "percentage": round(float(bilstm_probs[i]) * 100, 2)
          }
          for i in top_indices
      ]
      
      Same for CNN.
   
   g) Determine winner per case:
      Compare top-1 probability: which model is more confident?
      winner = "BiLSTM" if bilstm_probs.max() > cnn_probs.max() else "CNN"
   
   h) Return full result dict:
      {
        "input_text": clean_text,
        "extracted_symptoms": symptoms_list,
        "bilstm": {
            "predictions": bilstm_results,
            "top_disease": bilstm_results[0]["disease"],
            "top_confidence": bilstm_results[0]["percentage"],
            "model_name": "BiLSTM"
        },
        "cnn": {
            "predictions": cnn_results,
            "top_disease": cnn_results[0]["disease"],
            "top_confidence": cnn_results[0]["percentage"],
            "model_name": "CNN"
        },
        "agreement": bilstm_results[0]["disease"] == cnn_results[0]["disease"],
        "winner": winner,
        "mode": input_mode
      }
```

---

## 📊 STEP 8 — Visualization / Charts

### 👉 Cursor Prompt #9 — charts.py

```
In visualization/charts.py, write all Plotly chart functions.
Every function returns a plotly Figure object.

1. disease_comparison_bar(bilstm_results: list, cnn_results: list) -> go.Figure:
   
   Create a grouped bar chart comparing BiLSTM vs CNN predictions.
   
   - X axis: top 5 disease names
   - Y axis: probability percentage (0-100)
   - Two bars per disease: one for BiLSTM (blue), one for CNN (orange)
   - Add value labels on top of each bar
   - Title: "Disease Prediction Comparison — BiLSTM vs CNN"
   - Legend showing model names
   - Hover tooltip showing exact probability
   
   EXPLAIN: This is the MAIN comparison chart. For each predicted disease,
   you can see side-by-side how confident each model is.

2. confidence_gauge(bilstm_conf: float, cnn_conf: float) -> go.Figure:
   
   Create two side-by-side gauge charts:
   Left gauge: BiLSTM top disease confidence
   Right gauge: CNN top disease confidence
   
   Use plotly Indicator with mode="gauge+number"
   Color ranges:
   - 0-40: red
   - 40-70: yellow  
   - 70-100: green
   
   EXPLAIN: Gauge shows how CERTAIN each model is about its top prediction.
   High gauge = model is confident. Low gauge = uncertain case.

3. training_loss_curves(bilstm_history: dict, cnn_history: dict) -> go.Figure:
   
   Create a 2x2 subplot:
   Top-left: BiLSTM Training Loss vs Validation Loss per epoch
   Top-right: CNN Training Loss vs Validation Loss per epoch
   Bottom-left: BiLSTM Training Accuracy vs Validation Accuracy
   Bottom-right: CNN Training Accuracy vs Validation Accuracy
   
   Use different line styles: solid for train, dashed for validation
   Colors: BiLSTM in blue shades, CNN in orange shades
   Title: "Model Training History Comparison"
   
   EXPLAIN: Loss curves show HOW WELL each model learned.
   If val_loss diverges from train_loss → overfitting.
   Both curves going down together → good training.

4. model_metrics_radar(bilstm_metrics: dict, cnn_metrics: dict) -> go.Figure:
   
   Create a radar/spider chart comparing:
   Metrics: [Accuracy, Precision, Recall, F1-Score]
   
   Two traces: BiLSTM and CNN
   Fill with semi-transparent color
   Title: "Model Performance Metrics — Radar Chart"
   
   EXPLAIN: Radar chart shows overall model quality at a glance.
   A model with all metrics high = balanced good model.
   If one metric is low = weakness in that area.

5. probability_heatmap(bilstm_probs: list, cnn_probs: list, 
                        disease_names: list) -> go.Figure:
   
   Create a heatmap with:
   - Rows: ["BiLSTM", "CNN"]
   - Columns: top 10 disease names
   - Values: probability percentages
   - Color scale: white (0%) to dark blue (100%)
   - Show values inside cells
   - Title: "Probability Distribution Heatmap"
   
   EXPLAIN: Heatmap shows full probability spread across top diseases.
   Dark cells = high probability. Light cells = low probability.
   When both models are dark for same disease = high agreement.

6. winner_summary_card(result: dict) -> go.Figure:
   
   Create a simple table/card showing:
   | Metric          | BiLSTM        | CNN           | Winner |
   |-----------------|---------------|---------------|--------|
   | Top Disease     | Diabetes      | Diabetes      | Tie    |
   | Confidence      | 87.5%         | 79.2%         | BiLSTM |
   | Agreement       | ✅ Yes        | ✅ Yes        |   -    |
   
   Use plotly Table trace with colored header
   Green highlight on winner column
   Title: "Model Comparison Summary"
```

---

## 🎨 STEP 9 — Streamlit UI (Full)

### 👉 Cursor Prompt #10 — app.py

```
In app.py, build a professional Streamlit web application.

1. Page config:
   st.set_page_config(
       page_title="Disease Predictor — BiLSTM vs CNN",
       page_icon="🏥",
       layout="wide"
   )

2. Custom CSS using st.markdown with <style> tags:
   - Dark navy header: background gradient #0d1b2a to #1b263b
   - White text on header
   - Card containers with subtle shadow and rounded corners
   - Tab styling: active tab in blue
   - Model name badges: BiLSTM in blue pill, CNN in orange pill
   - Winner badge in green
   - Metric cards with colored left border

3. Header section:
   Centered title: "🏥 Disease Predictor"
   Subtitle: "Deep Learning Analysis — BiLSTM vs CNN"
   Three info metrics in columns:
   - "🧠 Models: 2 (BiLSTM + CNN)"
   - "🦠 Diseases: 132"
   - "📊 Dataset: Kaggle Symptom Dataset"

4. Sidebar:
   - App title and description
   - "📖 How It Works" expander with steps
   - "🧬 About the Models" expander:
     * BiLSTM: reads text forward and backward using LSTM memory
     * CNN: detects local symptom patterns using convolution filters
   - "⚙️ Settings" section:
     * Slider: "Number of diseases to show" (3-10, default 5)
     * Toggle: "Show training history charts"
     * Toggle: "Show probability heatmap"
   - Warning disclaimer

5. Main input area — Two tabs:
   
   Tab 1: "🤒 Symptom Analysis"
   - Large text area for symptom description
   - Placeholder with example symptom text
   - Analyze button
   
   Tab 2: "💊 Prescription Analysis"
   - Large text area for prescription input
   - Placeholder with example prescription
   - Note: "Drug names will be mapped to associated symptoms"
   - Analyze button
   
   Tab 3: "📚 Model Training Info"
   - Show training history charts (loaded from saved file)
   - Show model metrics comparison
   - Show model architecture summary

6. Results section (shown after analysis):
   
   a) Top summary row — 4 metric cards:
      - BiLSTM Top Disease (with blue badge)
      - BiLSTM Confidence %
      - CNN Top Disease (with orange badge)  
      - CNN Confidence %
   
   b) Agreement indicator:
      If both models agree → "✅ Both models agree: [Disease Name]" (green)
      If disagree → "⚠️ Models disagree — review results carefully" (yellow)
   
   c) Winner banner:
      "🏆 More Confident Model: [BiLSTM/CNN] ([X]% vs [Y]%)"
      Style as a colored banner
   
   d) Tabs for charts:
      
      Tab: "📊 Comparison Chart"
      → Show disease_comparison_bar chart (full width)
      → Explain below: "Higher bar = model is more confident about this disease"
      
      Tab: "🎯 Confidence Gauges"
      → Show confidence_gauge chart
      → Show winner_summary_card table below
      
      Tab: "🔥 Probability Heatmap"
      → Show probability_heatmap (if enabled in sidebar)
      → Explain: "Darker = higher probability"
      
      Tab: "📉 Training History"
      → Show training_loss_curves (if enabled in sidebar)
      → Show model_metrics_radar chart
   
   e) Detailed predictions expandable section:
      Two columns — BiLSTM | CNN
      
      For each column, show ranked list:
      1. 🏥 Disease Name ████████████ 87.5%
      2. 🏥 Disease Name ████████     65.2%
      ...
      
      Use st.progress for bars
      Color code: >70% green, 40-70% yellow, <40% red
   
   f) Extracted symptoms chips:
      Show each extracted symptom as a colored tag/badge
      "fever" "headache" "body pain" — pill style tags

7. Load model with st.cache_resource:
   @st.cache_resource
   def load_model_manager():
       manager = ModelManager()
       manager.load_all()
       return manager
   
   Show spinner while loading: "⏳ Loading AI models..."
   If models not found, show clear error:
   "⚠️ Models not trained yet. Please run: python train.py"

8. Error handling throughout with friendly messages.
```

---

## 🏋️ STEP 10 — Train Script

### 👉 Cursor Prompt #11 — train.py

```
Create train.py in the root folder — the main training script.

This script trains both BiLSTM and CNN models and saves everything needed.

1. Print a banner:
   print("=" * 60)
   print("  Disease Predictor — Model Training")
   print("  Training BiLSTM + CNN on Kaggle Dataset")
   print("=" * 60)

2. Initialize Trainer
3. Call trainer.prepare_data() — print data stats
4. Call trainer.build_models() — print architecture
5. Train BiLSTM with timing:
   import time
   start = time.time()
   bilstm_history = trainer.train_bilstm()
   bilstm_time = time.time() - start
   print(f"BiLSTM training time: {bilstm_time:.1f} seconds")

6. Train CNN with timing:
   start = time.time()
   cnn_history = trainer.train_cnn()
   cnn_time = time.time() - start

7. Evaluate both:
   results = evaluator.compare_both(X_test, y_test)
   
   Print comparison table:
   print("\n" + "=" * 50)
   print("MODEL COMPARISON RESULTS")
   print("=" * 50)
   print(f"{'Metric':<15} {'BiLSTM':>10} {'CNN':>10} {'Winner':>10}")
   print("-" * 50)
   print(f"{'Accuracy':<15} {bilstm_acc:>9.2%} {cnn_acc:>9.2%} {winner:>10}")
   print(f"{'F1-Score':<15} {bilstm_f1:>9.2%} {cnn_f1:>9.2%} {winner:>10}")
   print(f"{'Train Time':<15} {bilstm_time:>8.1f}s {cnn_time:>8.1f}s {'CNN':>10}")

8. Save training history to data/training_history.pkl for UI to display

9. Print final message:
   print("\n✅ Training complete!")
   print("Run the app with: streamlit run app.py")
```

---

## 🧪 STEP 11 — Test Cases

### 👉 Cursor Prompt #12 — test_models.py

```
Create test_models.py in root folder — test script for 5 disease scenarios.

Load ModelManager and run these 5 test cases. 
Print full results for each including both model predictions.

Test 1 — Respiratory:
input_text = "fever sore throat runny nose body ache fatigue cough"

Test 2 — Diabetes:
input_text = "frequent urination excessive thirst blurred vision fatigue weight loss"

Test 3 — Heart/Cardiac:
input_text = "chest pain shortness of breath dizziness sweating left arm pain"

Test 4 — Gastric:
input_text = "stomach pain nausea vomiting diarrhea loss of appetite bloating"

Test 5 — Neurological:
input_text = "severe headache vision changes confusion memory loss dizziness"

For each test case print:
- Input symptoms
- BiLSTM top 3 predictions with percentages
- CNN top 3 predictions with percentages  
- Which model is more confident
- Do both models agree?

Format output cleanly with separators.
```

---

## ✅ STEP 12 — Final Review

### 👉 Cursor Prompt #13 — Final Check

```
Review the complete disease_dl_predictor/ project:

1. Check all Keras imports are from tensorflow.keras (not standalone keras)
2. Verify BiLSTM model uses Bidirectional(LSTM()) correctly
3. Verify CNN model uses Functional API with parallel Conv1D branches
4. Check model save/load uses correct .h5 format
5. Verify tokenizer and label_encoder are saved and loaded with pickle correctly
6. Check charts.py returns go.Figure objects (not shows them with .show())
7. Verify st.cache_resource is used for ModelManager in app.py
8. Check all file paths use os.path.join for cross-platform compatibility
9. Verify training/data_loader.py handles NaN symptom columns correctly
10. Check train.py saves training_history.pkl after training

List every issue with: filename → line → problem → fix
Apply all fixes.
```

---

## 🔥 KEY CONCEPT REFERENCE CARD

```
WHY BiLSTM vs CNN?

BiLSTM:
"I have severe chest pain and shortness of breath"
  ← reads backward ←                    → reads forward →
  Captures: "shortness of breath" relates to earlier "chest pain"
  GOOD FOR: Long symptom descriptions, order-dependent patterns

CNN (with multiple kernel sizes):
kernel=2: [chest pain] [pain shortness] [shortness breath]
kernel=3: [chest pain shortness] [pain shortness breath]
kernel=4: [chest pain shortness breath]
  Each kernel captures different N-gram patterns
  GOOD FOR: Local symptom patterns, fast inference

WHEN BiLSTM WINS: Complex descriptions with context dependencies
WHEN CNN WINS: Short keyword-style inputs, faster prediction

Both trained on same data → fair comparison
Both predict same 132 diseases → same output space
```

---

## 📋 Prompts Order Summary

| # | Prompt | Creates |
|---|---|---|
| 1 | Scaffold | All folders + empty files |
| 2 | Requirements + Config | requirements.txt + settings.py |
| 3 | Data Loader + Setup | training/data_loader.py + setup.py |
| 4 | BiLSTM Model | models/bilstm_model.py ⭐ |
| 5 | CNN Model | models/cnn_model.py ⭐ |
| 6 | Trainer + Evaluator | training/trainer.py + evaluator.py |
| 7 | spaCy Extractors | nlp/preprocessor.py + symptom_extractor.py |
| 8 | Model Manager | models/model_manager.py |
| 9 | Charts | visualization/charts.py |
| 10 | Streamlit UI | app.py |
| 11 | Train Script | train.py |
| 12 | Test Script | test_models.py |
| 13 | Final Review | All fixes applied |

## 🚀 Run Order After Build

```bash
# 1. Install packages
pip install -r requirements.txt

# 2. Setup spaCy + folders
python setup.py

# 3. Download dataset from Kaggle and place in data/dataset.csv

# 4. Train both models (takes 10-20 mins on CPU)
python train.py

# 5. Launch the app
streamlit run app.py
```

> ⭐ BiLSTM and CNN files are the core — read all comments carefully in Cursor
