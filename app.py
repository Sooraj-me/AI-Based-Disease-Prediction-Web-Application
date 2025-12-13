import streamlit as st
import pandas as pd
from auth import login_user, register_user
from model import predict_disease
from database import c, conn

st.set_page_config(page_title="Disease Prediction", page_icon="🩺")

st.title("🩺 AI Disease Prediction System")

menu = ["Login", "Register", "Predict", "History"]
choice = st.sidebar.selectbox("Menu", menu)

df = pd.read_csv("data/symptoms.csv")
symptoms_list = df.columns[:-1]

# ---------------- REGISTER ----------------
if choice == "Register":
    st.subheader("Create Account")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Register"):
        register_user(u, p)
        st.success("Registration successful!")

# ---------------- LOGIN ----------------
elif choice == "Login":
    st.subheader("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(u, p)
        if user:
            st.session_state["user"] = u
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

# ---------------- PREDICT ----------------
elif choice == "Predict" and "user" in st.session_state:
    st.subheader("Select Symptoms")

    symptoms = {}
    for s in symptoms_list:
        symptoms[s] = st.checkbox(s)

    if st.button("Predict Disease"):
        probs = predict_disease(symptoms)
        disease_index = probs.argmax()
        disease_name = df["disease"].unique()[disease_index]

        st.success(f"Predicted Disease: **{disease_name}**")

        prob_df = pd.DataFrame({
            "Disease": df["disease"].unique(),
            "Probability": probs
        }).sort_values(by="Probability", ascending=False)

        st.bar_chart(prob_df.set_index("Disease"))

        c.execute(
            "INSERT INTO history VALUES (?,?,?)",
            (st.session_state["user"], str(symptoms), disease_name)
        )
        conn.commit()

# ---------------- HISTORY ----------------
elif choice == "History" and "user" in st.session_state:
    st.subheader("Prediction History")

    c.execute(
        "SELECT symptoms, disease FROM history WHERE username=?",
        (st.session_state["user"],)
    )

    rows = c.fetchall()
    for r in rows:
        st.write(f"**Symptoms:** {r[0]}")
        st.write(f"**Disease:** {r[1]}")
        st.markdown("---")

st.caption("⚠️ Educational purpose only. Not medical advice.")
