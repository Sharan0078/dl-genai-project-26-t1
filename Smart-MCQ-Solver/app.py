import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice
)

st.set_page_config(
    page_title="Smart MCQ Solver",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>

/* Hide Streamlit */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#111827,#1e293b);
}

/* Main container */
.block-container{
    padding-top:2rem;
    max-width:1100px;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
}

/* Progress bars */
.stProgress > div > div > div > div{
    background:linear-gradient(90deg,#2563eb,#22c55e);
}

</style>
""", unsafe_allow_html=True)

MODEL_NAME = "24f3004935/smart-mcq-solver"

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForMultipleChoice.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

with st.sidebar:

    st.title("🎓 Smart MCQ Solver")

    st.markdown("---")

    st.markdown("""
### 📘 Project

**IIT Madras**  
**Diploma in Data Science**  
**Deep Learning and Generative AI**

---

### 👨‍💻 Developed By

**E. Sharan Kumar**

**Roll No:** **24F3004935**

---

### 🚀 Features

- AI Powered MCQ Prediction
- DeBERTaV3 Large
- Confidence Scores
- PyTorch + Transformers
- Streamlit
""")

st.markdown("""
<h1 style='text-align:center;color:white;'>
🎓 Smart MCQ Solver
</h1>

<p style='text-align:center;
font-size:20px;
color:#cbd5e1;'>

Artificial Intelligence Powered
Multiple Choice Question Answering

</p>

""", unsafe_allow_html=True)

st.divider()

question = st.text_area(
    "Question",
    height=120
)

col1,col2 = st.columns(2)

with col1:

    option_a = st.text_area(
        "Option A",
        height=120
    )

    option_c = st.text_area(
        "Option C",
        height=120
    )

    option_e = st.text_area(
        "Option E",
        height=120
    )

with col2:

    option_b = st.text_area(
        "Option B",
        height=120
    )

    option_d = st.text_area(
        "Option D",
        height=120
    )

st.write("")

predict = st.button(
    "Predict Answer"
)

if predict:

    if any(x.strip() == "" for x in [
        question,
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]):
        st.error("!!!! Please fill all the fields.")
        st.stop()

    # Build text exactly like training
    options = [
        option_a,
        option_b,
        option_c,
        option_d,
        option_e
    ]

    inputs = tokenizer(
        [question] * 5,
        options,
        truncation=True,
        max_length=512,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt"
    )

    inputs = {
        k: v.unsqueeze(0)
        for k, v in inputs.items()
    }

    with st.spinner("Thinking..."):

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    labels = ["A", "B", "C", "D", "E"]

    prediction = torch.argmax(probabilities).item()

    st.markdown("---")

    st.markdown(f"""
    <div style="
    background:linear-gradient(135deg,#2563eb,#22c55e);
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 10px 30px rgba(0,0,0,0.3);
    ">

    <h3>Prediction</h3>

    <h1 style="font-size:70px;margin:0;">
    {labels[prediction]}
    </h1>

    <p style="font-size:20px;">
    Confidence: {probabilities[prediction].item()*100:.2f}%
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.subheader("📊 Confidence Scores")

    for label, score in zip(labels, probabilities):

        percentage = score.item() * 100

        st.write(f"### Option {label}")

        st.progress(float(score))

        st.write(f"**{percentage:.2f}%**")

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;
                color:#94a3b8;
                font-size:15px;
                padding:20px;">

    🎓 <b>IIT Madras - Diploma in Data Science</b><br>
    <b>Deep Learning for Generative AI Project</b><br><br>

    Developed by <b>E. Sharan Kumar</b><br>
    Roll No: <b>24F3004935</b>

    </div>
    """,
    unsafe_allow_html=True
)
