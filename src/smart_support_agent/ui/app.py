import json

import requests
import streamlit as st

# Constants
API_URL = "http://localhost:8000/api"

# Page config
st.set_page_config(
    page_title="Smart Support Agent",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Smart Support Agent")
st.markdown("""
This AI-powered system helps analyze and respond to customer support tickets.
Simply enter your support ticket content below, and the AI will:
- Categorize the ticket
- Assign priority
- Suggest an initial response
""")

# Input section
with st.container():
    ticket_content = st.text_area(
        "Enter support ticket content:",
        height=150,
        placeholder="Type or paste your support ticket content here..."
    )
    
    if st.button("Process Ticket", type="primary"):
        if ticket_content:
            try:
                # Show processing message
                with st.spinner("Processing ticket..."):
                    # Make API request
                    response = requests.post(
                        f"{API_URL}/process-ticket",
                        json={"content": ticket_content}
                    )
                    response.raise_for_status()
                    result = response.json()
                
                # Display results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Ticket Analysis")
                    st.info(f"Category: {result['category']}")
                    st.warning(f"Priority: {result['priority']}")
                    st.success(f"Ticket ID: {result['id']}")
                
                with col2:
                    st.subheader("Suggested Response")
                    st.text_area(
                        "Response:",
                        value=result['suggested_response'],
                        height=200,
                        disabled=True
                    )
                    
                    # Add copy button
                    if st.button("Copy Response"):
                        st.write("Response copied to clipboard! ✨")
                        st.session_state['clipboard'] = result['suggested_response']
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error processing ticket: {str(e)}")
        else:
            st.warning("Please enter ticket content first.")
