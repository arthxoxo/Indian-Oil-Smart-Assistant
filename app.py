import streamlit as st
import pandas as pd
import google.generativeai as genai # Correct import for Gemini
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import time
from typing import Optional
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Indian Oil Smart Assistant",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/a/a3/Indian_Oil_Logo.svg", # Updated to Indian Oil SVG logo URL
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Indian Oil Palette:
     * Orange: #FF6B35 (primary, already used)
     * Dark Blue: #002060 (from logo)
     * Lighter Blue: #003399 (for gradient variation)
     * Light Grey: #f5f5f5 (for card backgrounds)
     * Dark Text: #222, #333
     * White: #FFFFFF
     */

    .main-header {
        background: linear-gradient(90deg, #FF6B35, #F7931E); /* Keep existing orange gradient */
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #002060 0%, #003399 100%); /* Dark blue gradient from logo */
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .user-message {
        background: linear-gradient(135deg, #002060 0%, #003399 100%); /* Dark blue gradient for user messages */
        color: white;
        margin-left: 2rem;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #F7931E 0%, #FF6B35 100%); /* Orange gradient for bot messages */
        color: white;
        margin-right: 2rem;
    }
    
    .feature-card {
        background: white; /* Default background for feature cards */
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #FF6B35; /* Indian Oil orange border */
        margin: 1rem 0;
    }

    /* Styling for h4 headers within feature cards */
    .feature-card h4 {
        font-size: 1.25rem; /* Slightly smaller font size for headers */
        color: #333; /* Darker color for better readability on light backgrounds */
        margin-top: 0; /* Remove default top margin */
        margin-bottom: 0.75rem; /* Add some space below the header */
    }

    /* Corrected: Text color for content inside feature cards is now explicitly set to black for readability. */
    .feature-card p, .feature-card ul, .feature-card li {
        color: #000; /* Set text color to black for readability */
    }

    /* New CSS for About section feature cards - slightly distinct background */
    .about-feature-card {
        background: #f5f5f5; /* A slightly softer off-white background */
        box-shadow: 0 6px 20px rgba(0,0,0,0.15); /* Slightly stronger shadow */
        transition: transform 0.2s ease-in-out; /* Add a subtle hover effect */
    }

    .about-feature-card:hover {
        transform: translateY(-3px); /* Lift on hover */
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B35, #F7931E); /* Keep existing orange gradient */
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4);
    }
    
    .search-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }

    /* The fixed disclaimer class is removed to avoid positioning conflicts */
</style>
""", unsafe_allow_html=True)

# --- CONFIGURE YOUR GOOGLE GEMINI API KEY HERE ---
# Ensure only Google Generative AI (Gemini) is configured.
# The previous OpenAI import and client setup lines have been removed.
genai.configure(api_key="AIzaSyAYMgBuQOIlvpOF3Zycv7OqzabdnssxgTg") # Replace with your actual Gemini API key

# --- Load Retail Outlets Data ---
@st.cache_data
def load_outlets() -> pd.DataFrame:
    """
    Loads retail outlet data from a CSV file.
    Uses st.cache_data to cache the DataFrame for performance.
    Handles potential errors during CSV loading.
    """
    try:
        # Read as standard CSV with proper quote handling
        # Ensure 'tabula-Cash@PoS-UPDATED-19-11-16.csv' is in the same directory as app.py
        df = pd.read_csv("tabula-Cash@PoS-UPDATED-19-11-16.csv", quotechar='"', encoding='utf-8')
        # Clean up the data: strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        # Remove any completely empty rows
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}. Please ensure 'tabula-Cash@PoS-UPDATED-19-11-16.csv' is in the correct directory.")
        return pd.DataFrame()

outlets_df: pd.DataFrame = load_outlets()

# --- Helper Functions ---
def search_outlets(state: Optional[str] = None, district: Optional[str] = None, bank: Optional[str] = None) -> pd.DataFrame:
    """
    Searches the outlets DataFrame based on provided state, district, and bank filters.
    Filters are case-insensitive and support partial matches if exact match is not found.
    """
    df = outlets_df.copy()

    # Ensure df is a DataFrame before proceeding
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Filter by state first if provided
    if state and state != "All States":
        state_col = next((col for col in df.columns if 'STATE' in col.upper()), None)
        if state_col:
            # Try exact match first
            mask = df[state_col].fillna('').astype(str).str.lower() == state.lower()
            if not df[mask].empty:
                df = df[mask]
            else: # Fallback to contains if no exact match
                df = df[df[state_col].fillna('').astype(str).str.lower().str.contains(state.lower(), na=False)]

    # Then filter by district if provided
    if district and district != "All Districts":
        district_col = next((col for col in df.columns if 'DISTRICT' in col.upper()), None)
        if district_col:
            # Try exact match first
            mask = df[district_col].fillna('').astype(str).str.lower() == district.lower()
            if not df[mask].empty:
                df = df[mask]
            else: # Fallback to contains if no exact match
                df = df[df[district_col].fillna('').astype(str).str.lower().str.contains(district.lower(), na=False)]

    # Filter by bank if provided
    if bank and bank != "All Banks":
        bank_col = next((col for col in df.columns if 'BANK' in col.upper()), None)
        if bank_col:
            # Try exact match first
            mask = df[bank_col].fillna('').astype(str).str.lower() == bank.lower()
            if not df[mask].empty:
                df = df[mask]
            else: # Fallback to contains if no exact match
                df = df[df[bank_col].fillna('').astype(str).str.lower().str.contains(bank.lower(), na=False)]
    
    return df

def is_outlet_query(query: str) -> bool:
    """
    Checks if a user query is likely related to searching for outlets.
    """
    keywords = ["outlet", "retail", "station", "pump", "location", "where", "find", "near", "around"]
    return any(word in query.lower() for word in keywords)

def extract_district_state(query: str) -> tuple[Optional[str], Optional[str]]:
    """
    Attempts to extract district and state names from a user query
    by matching against unique values in the outlets DataFrame.
    Prioritizes exact matches, then partial matches.
    """
    district_col = None
    state_col = None

    # Find the actual column names for 'DISTRICT' and 'STATE'
    for col in outlets_df.columns:
        if 'DISTRICT' in col.upper():
            district_col = col
        elif 'STATE' in col.upper():
            state_col = col

    query_lower = query.lower()
    found_district = None
    found_state = None

    # Extract district
    if district_col and not outlets_df[district_col].empty:
        unique_districts = outlets_df[district_col].dropna().astype(str).str.strip().str.lower().unique()
        # Try exact match
        for district in unique_districts:
            if district and district in query_lower.split(): # Check for whole word match
                found_district = outlets_df[district_col][outlets_df[district_col].astype(str).str.lower() == district].iloc[0]
                break
        if not found_district: # Try partial match if no exact whole word match
            for district in unique_districts:
                if district and district in query_lower:
                    found_district = outlets_df[district_col][outlets_df[district_col].astype(str).str.lower() == district].iloc[0]
                    break

    # Extract state
    if state_col and not outlets_df[state_col].empty:
        unique_states = outlets_df[state_col].dropna().astype(str).str.strip().str.lower().unique()
        # Try exact match
        for state in unique_states:
            if state and state in query_lower.split(): # Check for whole word match
                found_state = outlets_df[state_col][outlets_df[state_col].astype(str).str.lower() == state].iloc[0]
                break
        if not found_state: # Try partial match if no exact whole word match
            for state in unique_states:
                if state and state in query_lower:
                    found_state = outlets_df[state_col][outlets_df[state_col].astype(str).str.lower() == state].iloc[0]
                    break

    return found_district, found_state

def create_state_distribution_chart():
    """
    Creates a horizontal bar chart showing the top 10 states by number of outlets.
    """
    if 'STATE' in outlets_df.columns and not outlets_df['STATE'].empty:
        state_counts = outlets_df['STATE'].value_counts().head(10)
        fig = px.bar(
            x=state_counts.values,
            y=state_counts.index,
            orientation='h',
            title="Top 10 States by Number of Outlets",
            labels={'x': 'Number of Outlets', 'y': 'State'},
            color=state_counts.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400, showlegend=False) # Hide color legend for single series
        return fig
    return None

def create_bank_distribution_chart():
    """
    Creates a pie chart showing the distribution of outlets by partner bank.
    """
    if 'BANK' in outlets_df.columns and not outlets_df['BANK'].empty:
        bank_counts = outlets_df['BANK'].value_counts()
        fig = px.pie(
            values=bank_counts.values,
            names=bank_counts.index,
            title="Distribution by Bank",
            hole=0.4 # Creates a donut chart
        )
        fig.update_layout(height=400)
        return fig
    return None

# --- Main App ---
def main():
    """
    Main function to run the Streamlit application.
    Manages navigation, displays dashboards, search, chat, and analytics sections.
    """
    # Placing the disclaimer here ensures it is at the very top of the app content
    # and scrolls with the rest of the page.
    st.markdown("""
    <p><b>Disclaimer:</b> All information, data, and resources used in this report have been sourced from publicly available materials on the internet. This report is created solely for academic and educational purposes as part of an internship project. It is not intended for public distribution, commercial use, or to represent any official stance of Indian Oil Corporation Limited or any other organization.</p>
    """, unsafe_allow_html=True)
    
    # This block now renders the Indian Oil Logo alongside the main title
    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 10px;">'
        f'<img src="https://upload.wikimedia.org/wikipedia/commons/a/a3/Indian_Oil_Logo.svg" alt="Indian Oil Logo" style="height: 40px;">'
        f'<h1>Indian Oil Smart Assistant</h1>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="main-header"><h1>Welcome to Indian Oil Smart Assistant!</h1></div>', unsafe_allow_html=True)

    # --- Navigation State Setup ---
    menu_options = ["🏠 Dashboard", "🔍 Search Outlets", "💬 Chat Assistant", "📊 Analytics", "ℹ️ About"]
    if "page" not in st.session_state:
        st.session_state.page = menu_options[0]

    # If a quick action was triggered from the dashboard, update the selected page
    page_map = {
        "search": "🔍 Search Outlets",
        "chat": "💬 Chat Assistant",
        "analytics": "📊 Analytics"
    }
    if "go_to_page" in st.session_state:
        st.session_state.page = page_map.get(st.session_state.go_to_page, menu_options[0])
        del st.session_state.go_to_page # Clear the trigger

    # Add sidebar navigation menu using option_menu
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        sidebar_selected = option_menu(
            menu_title=None, # No title for the menu itself
            options=menu_options,
            icons=["house", "search", "chat", "bar-chart", "info-circle"],
            menu_icon="cast",
            # Set default index based on current session state page
            default_index=menu_options.index(st.session_state.page) if st.session_state.page in menu_options else 0,
        )
        # Sync sidebar selection with main menu state
        if sidebar_selected != st.session_state.page:
            st.session_state.page = sidebar_selected
            st.rerun() # Rerun the app to navigate to the selected page

    selected = st.session_state.page # Use the page state from the sidebar for conditional rendering
    
    # --- Dashboard Section ---
    if selected == "🏠 Dashboard":
        st.markdown("### 🏠 Overview Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Total Outlets</h3>
                <h2>{len(outlets_df)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'STATE' in outlets_df.columns:
                unique_states = outlets_df['STATE'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🗺️ States Covered</h3>
                    <h2>{unique_states}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'DISTRICT' in outlets_df.columns:
                unique_districts = outlets_df['DISTRICT'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏘️ Districts</h3>
                    <h2>{unique_districts}</h2>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if 'BANK' in outlets_df.columns:
                unique_banks = outlets_df['BANK'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🏦 Partner Banks</h3>
                    <h2>{unique_banks}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Display charts
        st.markdown("---") # Separator
        col1, col2 = st.columns(2)
        
        with col1:
            state_chart = create_state_distribution_chart()
            if state_chart:
                st.plotly_chart(state_chart, use_container_width=True)
            else:
                st.info("State distribution chart not available. 'STATE' column not found or is empty.")
        
        with col2:
            bank_chart = create_bank_distribution_chart()
            if bank_chart:
                st.plotly_chart(bank_chart, use_container_width=True)
            else:
                st.info("Bank distribution chart not available. 'BANK' column not found or is empty.")
        
        # Quick Actions for navigation
        st.markdown("---") # Separator
        st.markdown("### ⚡ Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Find Nearest Outlet", use_container_width=True):
                st.session_state.go_to_page = "search"
                st.rerun()
        
        with col2:
            if st.button("💬 Ask Questions", use_container_width=True):
                st.session_state.go_to_page = "chat"
                st.rerun()
        
        with col3:
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.go_to_page = "analytics"
                st.rerun()
    
    # --- Search Outlets Section ---
    elif selected == "🔍 Search Outlets":
        st.markdown("### 🔍 Advanced Outlet Search")
        
        with st.container():
            st.markdown('<div class="search-box">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Populate state dropdown
                if 'STATE' in outlets_df.columns and not outlets_df['STATE'].empty:
                    states = sorted(outlets_df['STATE'].dropna().unique())
                    selected_state = st.selectbox("🏛️ Select State", ["All States"] + list(states))
                else:
                    selected_state = "All States"
                    st.warning("State data not available for filtering.")
            
            with col2:
                # Populate district dropdown, filtering by selected state
                if 'DISTRICT' in outlets_df.columns and not outlets_df['DISTRICT'].empty:
                    if selected_state and selected_state != "All States":
                        # Filter districts based on selected state
                        filtered_df_by_state = outlets_df[outlets_df['STATE'] == selected_state]
                        districts = sorted(filtered_df_by_state['DISTRICT'].dropna().unique())
                    else:
                        districts = sorted(outlets_df['DISTRICT'].dropna().unique())
                    selected_district = st.selectbox("🏘️ Select District", ["All Districts"] + list(districts))
                else:
                    selected_district = "All Districts"
                    st.warning("District data not available for filtering.")
            
            with col3:
                # Populate bank dropdown
                if 'BANK' in outlets_df.columns and not outlets_df['BANK'].empty:
                    banks = sorted(outlets_df['BANK'].dropna().unique())
                    selected_bank = st.selectbox("🏦 Select Bank", ["All Banks"] + list(banks))
                else:
                    selected_bank = "All Banks"
                    st.warning("Bank data not available for filtering.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Search button
        if st.button("🔍 Search Outlets", use_container_width=True):
            with st.spinner("Searching for outlets..."):
                time.sleep(1)  # Add a small delay for better UX
                
                # Prepare filters for the search_outlets function
                state_filter = selected_state if selected_state != "All States" else None
                district_filter = selected_district if selected_district != "All Districts" else None
                bank_filter = selected_bank if selected_bank != "All Banks" else None
                
                results = search_outlets(state=state_filter, district=district_filter, bank=bank_filter)
                
                if not results.empty:
                    st.success(f"✅ Found {len(results)} outlets!")
                    
                    # Define columns to display in the results table
                    display_cols = []
                    # Check for common relevant columns, prioritizing them
                    preferred_cols = ['NAME', 'LOCATION', 'ADDRESS', 'DISTRICT', 'STATE', 'BANK', 'PINCODE']
                    for col_name in preferred_cols:
                        if col_name.upper() in [c.upper() for c in results.columns]:
                            # Find the exact case-sensitive column name
                            actual_col_name = next(c for c in results.columns if c.upper() == col_name.upper())
                            display_cols.append(actual_col_name)
                    
                    # If no preferred columns found, display all columns
                    if not display_cols:
                        display_cols = results.columns.tolist()

                    st.dataframe(results[display_cols], use_container_width=True)
                    
                    # Download option for results
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"indian_oil_outlets_{state_filter or 'all'}_{district_filter or 'all'}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ No outlets found matching your criteria. Try broadening your search!")
    
    # --- Chat Assistant Section ---
    elif selected == "💬 Chat Assistant":
        st.markdown("### 💬 Intelligent Chat Assistant")
        
        # Initialize chat history in session state if not already present
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat input box
        user_input = st.text_input("💭 Ask me anything about Indian Oil or search for outlets:", key="user_input")
        
        col1, col2 = st.columns([1, 4]) # Layout for send and clear buttons
        with col1:
            send_button = st.button("🚀 Send", use_container_width=True)
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = [] # Clear history
                st.rerun() # Rerun to update the display
        
        # Process user input when send button is clicked or Enter is pressed in text_input
        if (send_button or user_input) and user_input:
            # Add user message to history
            st.session_state.chat_history.append(("user", user_input))
            
            # Check if the query is for outlets
            if is_outlet_query(user_input):
                with st.spinner("🔍 Searching for outlets..."):
                    district, state = extract_district_state(user_input)
                    results = search_outlets(state, district)
                    
                    if not results.empty:
                        response = f"🎯 Found {len(results)} Indian Oil retail outlets for you!"
                        st.session_state.chat_history.append(("bot", response))
                        
                        # Define columns to display for outlet search results in chat
                        display_cols = []
                        preferred_cols = ['NAME', 'LOCATION', 'ADDRESS', 'DISTRICT', 'STATE', 'BANK']
                        for col_name in preferred_cols:
                            if col_name.upper() in [c.upper() for c in results.columns]:
                                actual_col_name = next(c for c in results.columns if c.upper() == col_name.upper())
                                display_cols.append(actual_col_name)
                        
                        if display_cols:
                            st.session_state.chat_history.append(("bot_df", results[display_cols].head(5))) # Show top 5 results
                        else:
                            st.session_state.chat_history.append(("bot_df", results.head(5)))
                    else:
                        response = "😔 Sorry, I couldn't find any outlets matching your query. Try searching for a different location or being more specific!"
                        st.session_state.chat_history.append(("bot", response))
            else:
                # If not an outlet query, use the Gemini model
                with st.spinner("🤖 Thinking..."):
                    try:
                        # Initialize the Gemini GenerativeModel with the corrected model name
                        # Changed model name to gemini-1.5-flash-latest for better compatibility based on previous debug info
                        model = genai.GenerativeModel('gemini-1.5-flash-latest')
                        # Generate content based on user input
                        response = model.generate_content(user_input)
                        # Extract text from the response
                        bot_reply = response.text if hasattr(response, "text") else str(response)
                    except Exception as e:
                        bot_reply = f"Error communicating with AI: {e}. Please try again later."
                    st.session_state.chat_history.append(("bot", bot_reply))
        
        # Display chat history with newest chats at the top
        st.markdown("### 💬 Chat History")
        # Use a container to make chat history scrollable if it gets long
        chat_container = st.container(height=400, border=True) 
        with chat_container:
            # Iterate through chat history in reverse order to show newest at top
            for entry in reversed(st.session_state.chat_history):
                if entry[0] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {entry[1]}
                    </div>
                    """, unsafe_allow_html=True)
                elif entry[0] == "bot":
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant:</strong> {entry[1]}
                    </div>
                    """, unsafe_allow_html=True)
                elif entry[0] == "bot_df":
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>🤖 Assistant (Outlets Found):</strong>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(entry[1], use_container_width=True) # Display DataFrame directly

    # --- Analytics Section ---
    elif selected == "📊 Analytics":
        st.markdown("### 📊 Data Analytics & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 State-wise Distribution")
            state_chart = create_state_distribution_chart()
            if state_chart:
                st.plotly_chart(state_chart, use_container_width=True)
            else:
                st.info("State distribution chart not available. 'STATE' column not found or is empty.")
        
        with col2:
            bank_chart = create_bank_distribution_chart()
            if bank_chart:
                st.plotly_chart(bank_chart, use_container_width=True)
            else:
                st.info("Bank distribution chart not available. 'BANK' column not found or is empty.")
        
        st.markdown("---") # Separator
        st.markdown("#### 📋 Detailed Statistics")
        
        # Display additional metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Outlets", int(len(outlets_df)))
            if 'STATE' in outlets_df.columns:
                st.metric("States Covered", int(outlets_df['STATE'].nunique()))
            else:
                st.metric("States Covered", "N/A")
        
        with col2:
            if 'DISTRICT' in outlets_df.columns:
                st.metric("Districts Covered", int(outlets_df['DISTRICT'].nunique()))
            else:
                st.metric("Districts Covered", "N/A")
            if 'BANK' in outlets_df.columns:
                st.metric("Partner Banks", int(outlets_df['BANK'].nunique()))
            else:
                st.metric("Partner Banks", "N/A")
        
        with col3:
            # Display top 5 states by outlet count
            if 'STATE' in outlets_df.columns and not outlets_df['STATE'].empty:
                top_states = outlets_df['STATE'].value_counts().head(5)
                st.markdown("**🏆 Top 5 States:**")
                for state, count in top_states.items():
                    st.write(f"• {state}: {count} outlets")
            else:
                st.info("Top states data not available.")
    
    # --- About Section ---
    elif selected == "ℹ️ About":
        st.markdown("### ℹ️ About Indian Oil Smart Assistant")
        
        # Apply both feature-card and about-feature-card classes
        st.markdown("""
        <div class="feature-card about-feature-card">
            <h4>🎯 What We Do</h4>
            <p>Indian Oil Smart Assistant is your intelligent companion for discovering retail outlets, 
            getting information about Indian Oil products and services, and accessing real-time data 
            about our extensive network across India.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Apply both feature-card and about-feature-card classes
            st.markdown("""
            <div class="feature-card about-feature-card">
                <h4>🔍 Features</h4>
                <ul>
                    <li>Smart outlet search by location</li>
                    <li>Real-time data analytics</li>
                    <li>AI-powered chat assistant</li>
                    <li>Interactive visualizations</li>
                    <li>Export functionality</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Apply both feature-card and about-feature-card classes
            st.markdown("""
            <div class="feature-card about-feature-card">
                <h4>🚀 Technology</h4>
                <ul>
                    <li>Streamlit for web interface</li>
                    <li>Google Gemini for AI responses</li>
                    <li>Plotly for data visualization</li>
                    <li>Pandas for data processing</li>
                    <li>Modern UI/UX design</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Entry point for the Streamlit application
if __name__ == "__main__":
    main() # Call the main function to run the app
