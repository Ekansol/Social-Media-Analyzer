# ============================================================
# SMART SOCIAL MEDIA ANALYZER — MAIN RUNNER
# Run this file to execute the full project pipeline
# ============================================================
#
# FILES IN THIS PROJECT:
#   main.py                           ← this file (run this)
#   part1_sentiment_analysis.py       ← BiLSTM sentiment classifier
#   part2_cnn_image_classification.py ← ResNet50 transfer learning
#   part3_autoencoder_fake_detection.py ← LSTM autoencoder (anomaly)
#   part4_gan_image_generation.py     ← GAN image generator
#   part5_nltk_nlp_pipeline.py        ← full NLTK preprocessing
#
# HOW TO RUN (in terminal / Jupyter / Colab):
#   python main.py
#
# OR run each part individually:
#   python part1_sentiment_analysis.py
#   python part2_cnn_image_classification.py
#   etc.
#
# REQUIREMENTS (install once):
#   pip install tensorflow keras numpy pandas matplotlib scikit-learn nltk Pillow
#
# ============================================================

print("""
╔══════════════════════════════════════════════════════════╗
║        SMART SOCIAL MEDIA ANALYZER — Deep Learning       ║
║                                                          ║
║  Part 1 → Sentiment Analysis    (BiLSTM, Unit IV)        ║
║  Part 2 → Image Classification  (CNN+ResNet, Unit III)   ║
║  Part 3 → Fake Post Detection   (Autoencoder, Unit V)    ║
║  Part 4 → Image Generation      (GAN, Unit V)            ║
║  Part 5 → NLP Pipeline          (NLTK, Labs 12-20)       ║
╚══════════════════════════════════════════════════════════╝
""")

# ── Choose which parts to run ────────────────────────────────
RUN_SENTIMENT   = True
RUN_CNN         = True
RUN_AUTOENCODER = True
RUN_GAN         = True
RUN_NLP         = True

if RUN_SENTIMENT:
    print("\n" + "▶" * 3 + " PART 1: SENTIMENT ANALYSIS " + "▶" * 3)
    exec(open('part1_sentiment_analysis.py').read())

if RUN_CNN:
    print("\n" + "▶" * 3 + " PART 2: IMAGE CLASSIFICATION " + "▶" * 3)
    exec(open('part2_cnn_image_classification.py').read())

if RUN_AUTOENCODER:
    print("\n" + "▶" * 3 + " PART 3: FAKE POST DETECTION " + "▶" * 3)
    exec(open('part3_autoencoder_fake_detection.py').read())

if RUN_GAN:
    print("\n" + "▶" * 3 + " PART 4: GAN IMAGE GENERATION " + "▶" * 3)
    exec(open('part4_gan_image_generation.py').read())

if RUN_NLP:
    print("\n" + "▶" * 3 + " PART 5: NLP PIPELINE " + "▶" * 3)
    exec(open('part5_nltk_nlp_pipeline.py').read())

print("\n✓ All parts completed successfully!")
print("  Output files: *.h5 (models), *.png (plots), gan_samples/ (images)")
