import re
import streamlit as st
import pandas as pd
from utils.helper import *

st.set_page_config(layout = "wide")
st.title("Dual Agent Interview Simulation")


with st.sidebar:
    with st.expander("Instruction Manual"):
        st.markdown("""
        # Agent Interview Simulation App

        Welcome to the **Agent Interview Simulation** app! This Streamlit application allows you to simulate a technical interview using Meta's Llama3 model and save the generated code.

        ## Features

        ### Instruction Manual

        **How to Use:**
        1. **Input:** Type a topic into the input labeled "Enter a Topic" and a number of interview questions into the input box labeled "Number of Questions".
        2. **Submit:** Press the "Submit" button to start the simulation.
        3. **Chat History:** View how the model learns over time and save the final Python scripts.


        **Credits:**
        - Based on the [Duel Agent Simulation](https://huggingface.co/eagle0504) and the [WYN-Agent](https://huggingface.co/eagle0504) projects by Yiqiao Yin.
        """)

    # Text input
    user_topic = st.text_input("Enter a topic", "Data Science")

    # Number of questions
    number_of_questions = st.number_input("Enter a number of interview questions (between 1 and 5)", min_value = 1, max_value = 5, value = 3, step = 1)

    # Add a button to submit
    submit_button = st.button("Run Simulation!")

    # Add a button to clear the session state
    if st.button("Clear Session"):
        st.session_state.messages = []
        st.rerun()


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Create agents
interviewer = LlamaBot(role_context = "You are an interviewer performing a technical interview of a prospective hire for a role requiring Python.")
interviewee = LlamaBot(role_context = "You are an inexperienced programmer practicing technical interviews for a role requiring Python code.")
judge = LlamaBot(role_context = "You are an experienced programmer tasked with evaluating and providing feedback on technical interview questions and Python code.")

list_of_questions = []
list_of_final_answers = []

if submit_button:
    # Initiatization
    prompt = f"Ask a programming question about this topic: {user_topic} that can be answered with a Python function."

    # Display user message in chat message container
    # Default: user = interviewee, assistant = interviewer
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    while len(list_of_questions) < number_of_questions:
        
        # Interview asks a question
        question = interviewer.call_llama(prompt +
                                  """
                                  Only provide a question, example input, and example output. Do not provide hints or example solutions.
                                  Make sure to format the question using the bolded header "Question: "
                                  """)
        question_shortened = re.search(r"(\*\*Question:.*)", question).group(1)

        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(question)
        st.session_state.messages.append({"role": "assistant", "content": question})

        # Clear the interviewee and judge conversation histories to start fresh for each question
        interviewee.clear_history()
        judge.clear_history()

        # Initialize variables to capture data about learning process
        passed = 0
        iter = 0
        list_of_iters = []
        list_of_answers = []
        list_of_judge_comments = []
        list_of_ratings = []

        
        while not passed and iter < 5:
            # Interviewee attempts an answer
            if iter < 1:
                answer = interviewee.call_llama(
                    f"""
                        Answer the question: {question} in a poor way.
                        Explain how you would solve the problem verbally using casual language, then provide a draft of a Python function.
                        Do not state that this is a poor attempt or explain why the attempt is poor.
                    """
                )
                st.chat_message("user").markdown(answer)
                st.session_state.messages.append({"role": "user", "content": answer})
            else:
                answer = interviewee.call_llama(
                    f"""
                        Answer the question: {question} in a poor way.
                        Explain how you would solve the problem verbally using casual language, then provide a draft of a Python function.
                        Do not state that this is a poor attempt or explain why the attempt is poor.
                        You are trying to learn, so use {judge_comments} as feedback to improve your answer.
                    """
                )
                st.chat_message("user").markdown(answer)
                st.session_state.messages.append({"role": "user", "content": answer})
    
            # Judge thinks and advises but the thoughts are hidden
            judge_comments = judge.call_llama(
                f"""
                    The question is: {question}
                    The answer is: {answer}
                    Provide feedback on the answer based on solving the problem posed by the question.
                    Rate the answer from 1 to 5. Only use integer ratings. If there are at least 2 improvements identified, do not give a rating higher than 3.
                    Heavily penalize non-professional tone and a lack of efficiency, docstrings, type hints, or comments.
                    Be sure to provide the rating at the beginning of your response in the form - Rating: X/5
                """
            )
            judge.clear_history() # History needs to be cleared or else the judge uses old data for feedback
        

            # Collect all responses
            rating_search = re.search(r"Rating: (\d)", judge_comments)
            rating = rating_search.group(1) if rating_search else 0
            passed = 1 if int(rating) >= 4 else 0
            list_of_iters.append(iter)
            list_of_answers.append(answer)
            list_of_judge_comments.append(judge_comments)
            list_of_ratings.append(rating)
            results_tab = pd.DataFrame({
                "Iter.": list_of_iters,
                "Answers": list_of_answers,
                "Judge Comments": list_of_judge_comments,
                "Ratings": list_of_ratings
            })
            
            # If the interviewee passed, save the final answer or save the best answer after 5 attempts
            if passed or iter == 4:
                list_of_questions.append(question_shortened)
                list_of_final_answers.append(interviewee.get_history()[-1]["content"])

            with st.expander(f"See responses"):
                st.table(results_tab)

            iter += 1
        
    # Show the final answers and save scripts to a single file
    code_snippets = []
    
    for question, final_answer in list(zip(list_of_questions, list_of_final_answers)):
        st.chat_message("user").markdown(f"""
        {question}
        
        **Answer:** {final_answer}
        """)

        code_snippets.append(isolate_python_code(final_answer))
    
    # Save scripts
    save_python_scripts(code_snippets)
