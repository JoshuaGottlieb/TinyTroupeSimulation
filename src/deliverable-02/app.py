import streamlit as st
import pandas as pd
from utils.helper import *

st.set_page_config(layout = "wide")
st.title("Duel Agent Simulation 🦙🦙")


with st.sidebar:
    with st.expander("Instruction Manual"):
        st.markdown("""            # Agient Simulation App        """)

    # Text input
    user_topic = st.text_input("Enter a topic", "Data Science")

    # Add a button to submit
    submit_button = st.button("Run Simulation!")

    # Add a button to clear the session state
    if st.button("Clear Session"):
        st.session_state.messages = []
        st.experimental_rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Create agents
interviewer = call_llama
interviewee = call_llama
judge = call_llama


# React to user input
iter = 0
list_of_iters = []
list_of_questions = []
list_of_answers = []
list_of_judge_comments = []
list_of_passes = []

if submit_button:
    # Initiatization
    prompt = f"Ask a question about this topic: {user_topic}"

    # Display user message in chat message container
    # Default: user = interviewee, assistant = interviewer
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    while True:

        # Interview asks a question
        question = interviewer(prompt)

        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(question)
        st.session_state.messages.append({"role": "assistant", "content": question})

        # Interviewee attempts an answer
        if iter < 5:
            answer = interviewee(
                f"""
                    Answer the question: {question} in a mediocre way
                    Because you are an inexperienced interviewee.
                """
            )
            st.chat_message("user").markdown(answer)
            st.session_state.messages.append({"role": "user", "content": answer})
        else:
            answer = interviewee(
                f"""
                    Answer the question: {question} in a mediocre way
                    Because you are an inexperienced interviewee but you really want to learn,
                    so you learn from the judge comments: {judge_comments}
                """
            )
            st.chat_message("user").markdown(answer)
            st.session_state.messages.append({"role": "user", "content": answer})

        # Judge thinks and advises but the thoughts are hidden
        judge_comments = judge(
            f"""
                The question is: {question}
                The answer is: {answer}
                Provide feedback and rate the answer from 1 to 10 while 10 being the best and 1 is the worst. 
            """
        )
        

        # Collect all responses
        passed_or_not = 1 if '8' in judge_comments else 0
        list_of_iters.append(iter)
        list_of_questions.append(question)
        list_of_answers.append(answer)
        list_of_judge_comments.append(judge_comments)
        list_of_passes.append(passed_or_not)
        results_tab = pd.DataFrame({
            "Iter.": list_of_iters,
            "Questions": list_of_questions,
            "Answers": list_of_answers,
            "Judge Comments": list_of_judge_comments,
            "Passed": list_of_passes
        })

        with st.expander("See explanation"):
            st.table(results_tab)

        # Stopping rule
        if '8' in judge_comments:
            break

        # Checkpoint
        iter += 1