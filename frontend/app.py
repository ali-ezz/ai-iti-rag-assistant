import streamlit as st
from api_client import AssistantAPIClient, AssistantAPIError

st.set_page_config(
    page_title="AI-ITI Study Assistant",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 880px; padding-top: 2.5rem;}
    .source-card {background:#f5f7fb; border-left:4px solid #6c63ff;
                  border-radius:8px; padding:.7rem .9rem; margin:.35rem 0;}
    [data-testid="stChatMessage"] {border-radius:14px; padding:.25rem .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 AI-ITI Study Assistant")
st.caption("A grounded RAG assistant for Python, Computer Vision, NLP, YOLO, and Transformers")

with st.expander("How this assistant works"):
    st.write(
        "Your question is matched against course material stored in Chroma. "
        "Only the retrieved sections are sent to a cloud-hosted Ollama model, "
        "and the answer includes the source sections used."
    )

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me something about the AI-ITI course material.",
            "sources": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.markdown(f'<div class="source-card">{source}</div>', unsafe_allow_html=True)

question = st.chat_input("Ask about OOP, computer vision, NLP, YOLO, or transformers…")
if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner(
        "Searching the course material and generating a grounded answer…"
    ):
        try:
            result = AssistantAPIClient().ask(question)
            answer = result["answer"]
            sources = result.get("sources", [])
            st.markdown(answer)
            if sources:
                with st.expander(f"Sources ({len(sources)})", expanded=True):
                    for source in sources:
                        st.markdown(
                            f'<div class="source-card">{source}</div>',
                            unsafe_allow_html=True,
                        )
        except AssistantAPIError as exc:
            answer = str(exc)
            sources = []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
