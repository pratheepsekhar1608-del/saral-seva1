# Technical Design Document (TDD) - Saral Seva

## 1. System Overview
Saral Seva provides an offline-first, privacy-preserving pipeline to extract citizen data from physical identity cards and automatically populate web registration forms without transmitting personal data to third-party cloud APIs.

---

## 2. Architectural Diagram

```text
+-------------------------------------------------------------------------------+
|                                USER INTERFACE                                 |
|                         (Streamlit / Local Web UI)                            |
+-------------------------------------------------------------------------------+
         | (1. Video Feed / Snap)                     ^ (5. Auto-Fill Data)
         v                                            |
+-----------------------+                    +------------------+
|     OpenCV Layer      |                    |  Form Population |
| - Video Stream Capture|                    | - Streamlit State|
| - Frame Preprocessing |                    | - Target Schema  |
+-----------------------+                    +------------------+
         | (2. Preprocessed Image)                    ^
         v                                            |
+-----------------------+                    +------------------+
|    PaddleOCR Engine   |                    |   Ollama LLM     |
| - Text Detection      |                    | - Local Inference|
| - Multilingual Recog  |                    | - JSON Extraction|
+-----------------------+                    +------------------+
         | (3. Raw Bounding Box Text)                 ^
         +--------------------------------------------+
                 (4. Prompt + Raw Strings)