# Test Cases & Verification Suite

## 1. Test Execution Matrix

| Test ID | Scenario | Input / Precondition | Expected Output | Status |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | Optical Document Capture | Live webcam feed with standard ID alignment | High-contrast snapshot captured without blur | Pass |
| **TC-02** | Multilingual OCR Recognition | Identity card image containing standard text | PaddleOCR outputs accurate bounding box tokens (>90% accuracy) | Pass |
| **TC-03** | Local LLM Schema Structuring | Raw token list fed to local Ollama instance | Clean, valid JSON mapping fields (`name`, `dob`, `id_number`) | Pass |
| **TC-04** | Dynamic UI Pre-Filling | Validated JSON entity payload | Streamlit form fields pre-fill automatically | Pass |
| **TC-05** | Offline Air-Gapped Operation | Network disabled (Airplane Mode) | Inference completes locally with zero cloud API dependencies | Pass |

---

## 2. Sample Data Payloads

### Sample A: Standard Clean Record
**Raw OCR Input:**
```text
Government of India
Ramesh Kumar
DOB: 15/08/1990
Male
1234 5678 9012



#JSON output


{
  "full_name": "Ramesh Kumar",
  "dob": "1990-08-15",
  "gender": "Male",
  "id_number": "1234 5678 9012",
  "id_type": "Aadhaar"
}

SAMPLE 2





Plain text

Govt of Ind
S/O Ramasamy
K. Anand
D.O.B: 02-04-1985
Female
ABCDE1234F

JSON output
{
  "full_name": "K. Anand",
  "dob": "1985-04-02",
  "gender": "Female",
  "id_number": "ABCDE1234F",
  "id_type": "PAN"
}
