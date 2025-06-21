import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
from utils.auth import get_hashed_user_data, register_user, verify_user
from components.dashboard import show_dashboard
from components.prediction import show_prediction
from components.portfolio import show_portfolio


# Page config
st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="📊",
    layout="wide"
)

# Custom CSS with just the colors
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --main-purple: #2E0F5A;
        --light-purple: #3D1C66;
        --bg-purple: #F3E6FF;
    }
    
    .stButton>button {
        background-color: var(--main-purple);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--light-purple);
    }

    /* Custom CSS for the registration form */
    .auth-form {
        background-color: rgba(46, 15, 90, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'name' not in st.session_state:
    st.session_state['name'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'show_register' not in st.session_state:
    st.session_state['show_register'] = False

# Toggle between login and register
def toggle_register():
    st.session_state['show_register'] = not st.session_state['show_register']

# Main authentication section
if not st.session_state['authentication_status']:
    st.title("Welcome to Portfolio Optimization")
    
    if not st.session_state['show_register']:
        # Login Form
        with st.form("login_form"):
            st.subheader("🔐 Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Login")
            with col2:
                st.form_submit_button("Register", on_click=toggle_register)
            
            if submit and username and password:
                success, name = verify_user(username, password)
                if success:
                    st.session_state['authentication_status'] = True
                    st.session_state['name'] = name
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    else:
        # Registration Form
        with st.form("register_form"):
            st.subheader("📝 Register")
            
            # Username field with requirements
            st.markdown("""
            ##### Username Requirements:
            - Must be 3-20 characters long
            - Must start with a letter
            - Can only contain letters, numbers, and underscores
            - Must be unique
            """)
            new_username = st.text_input("Username*")
            
            # Password fields
            st.markdown("##### Password Requirements:")
            st.markdown("- Both passwords must match")
            new_password = st.text_input("Password*", type="password")
            confirm_password = st.text_input("Confirm Password*", type="password")
            
            # Other fields
            name = st.text_input("Full Name*")
            email = st.text_input("Email*")
            
            col1, col2 = st.columns(2)
            with col1:
                register = st.form_submit_button("Create Account")
            with col2:
                st.form_submit_button("Back to Login", on_click=toggle_register)
            
            if register:
                if not all([new_username, new_password, confirm_password, name, email]):
                    st.error("All fields are required")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(new_username, name, email, new_password)
                    if success:
                        st.success(message)
                        st.session_state['show_register'] = False
                        st.rerun()
                    else:
                        st.error(message)

else:
    st.sidebar.success(f"Welcome {st.session_state['name']}!")

    # Navigation with the purple theme
    selected_page = option_menu(
        menu_title="Main Menu",
        options=["🏠 Dashboard", "📈 Prediction", "💰 Portfolio Optimization"],
        icons=["house", "graph-up", "currency-exchange"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "background-color": "#2E0F5A",
                "padding": "1.2rem",
                "border-radius": "8px"
            },
            "icon": {
                "color": "white",
                "font-size": "1.1rem"
            },
            "nav-link": {
                "color": "white",
                "font-size": "1rem",
                "text-align": "left",
                "margin": "0.5rem",
                "--hover-color": "#3D1C66",
                "padding": "0.8rem",
                "border-radius": "6px"
            },
            "nav-link-selected": {
                "background-color": "#3D1C66",
                "font-weight": "bold"
            },
            "menu-title": {
                "color": "white",
                "font-size": "1.2rem",
                "font-weight": "bold",
                "margin-bottom": "1rem"
            }
        }
    )

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state['authentication_status'] = None
        st.session_state['name'] = None
        st.session_state['username'] = None
        st.rerun()

    # Load pages
    if selected_page == "🏠 Dashboard":
        show_dashboard()
    elif selected_page == "📈 Prediction":
        show_prediction()
    elif selected_page == "💰 Portfolio Optimization":
        show_portfolio()
