import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load datasets
final_financial_report = pd.read_csv('Final_Financial_Report.csv')
summary_financial_analysis = pd.read_csv('summary_financial_analysis.csv')

# Initialize the model with your API key
model = ChatGroq(model="mixtral-8x7b-32768", api_key='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')

# Generate the prompt structure including Assets, Liabilities, and Cash Flow from Operations
def create_prompt(final_report, summary_report):
    final_report_text = "\n".join([
        f"{row['Year']}, {row['Company']}: Total Revenue = ${row['Total Revenue']}M, Net Income = ${row['Net Income']}M, "
        f"Total Assets = ${row['Total Assets']}M, Total Liabilities = ${row['Total Liabilities']}M, "
        f"Cash Flow from Operations = ${row['Cash Flow from Operating Activities']}M"
        for _, row in final_report.iterrows()
    ])

    summary_report_text = "\n".join([
        f"Revenue Growth = {row['Revenue Growth (%)']}%, Net Income Growth = {row['Net Income Growth (%)']}%, "
        f"Assets Growth = {row['Assets Growth (%)']}%, Liabilities Growth = {row['Liabilities Growth (%)']}%, "
        f"Cash Flow from Operations Growth = {row['Cash Flow from Operations Growth (%)']}%"
        for _, row in summary_report.iterrows()
    ])

    prompt = f"""
    ### Final Financial Report:
    {final_report_text}

    ### Summary Financial Analysis:
    {summary_report_text}

    ### Query: {{user_query}}
    """
    return prompt

# Create a chat prompt template using the function
chat_prompt = ChatPromptTemplate.from_template(create_prompt(final_financial_report, summary_financial_analysis))

# Function to handle queries
def chatbot(query):
    # Format the prompt with the user query
    prompt = chat_prompt.format(user_query=query)

    # Generate response from the ChatGroq model
    response = model.invoke(prompt)

    # Return the response text
    return response.content

# Streamlit Interface
def main():
    st.title("Financial Chatbot")
    st.write("Ask any financial-related question based on the provided dataset.")

    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    # Initialize session state for user query
    if 'user_query' not in st.session_state:
        st.session_state.user_query = ''

    # Input box for the query at the top
    # Update input box to always reflect the current state
    user_query = st.text_input("Enter your query:", value=st.session_state.user_query)

    # Submit button next to the input box
    submit_button = st.button("Submit")

    # If user submits the query
    if submit_button:
        if user_query:
            with st.spinner('Fetching response...'):
                # Get the response from the chatbot
                response = chatbot(user_query)

                # Append the question and response to the conversation history
                st.session_state.conversation.append((user_query, response))

                # Clear the input field after submission
                st.session_state.user_query = ''  # Clear input box

                # Remove previous query display
                st.rerun()  # Ensure the UI reflects the cleared input
        else:
            st.write("Please enter a query.")

    # Display conversation history with the most recent responses at the top
    for q, a in reversed(st.session_state.conversation):
        with st.container():
            st.markdown(
                f"<div style='border: 1px solid red; border-radius: 5px; padding: 10px; margin-bottom: 10px;'>"
                f"<div style='border: 1px solid #007BFF; border-radius: 5px; padding: 10px; margin-bottom: 5px;'>"
                f"<strong>User:</strong> {q}</div>"
                f"<div style='border: 1px solid #28A745; border-radius: 5px; padding: 10px;'>"
                f"<strong>Response:</strong> {a}</div>"
                f"</div>", 
                unsafe_allow_html=True,
            )

if __name__ == '__main__':
    main()
