import streamlit as st
import datetime

# Inizializza il contatore se non esiste
if "counter" not in st.session_state: 
    st.session_state.counter = 0
    # st.session_state.last_updated = datetime.time(0,0)


def increment():
    st.session_state.counter += 1
    # st.session_state.last_updated = st.session_state.update_time

def decrement():
    st.session_state.counter -= 1
    # st.session_state.last_updated = st.session_state.update_time


# pages = {
#     "Azioni": [
#         st.Page("streamlit.py", title="Contatore"),
#         st.Page("nome_utente.py", title="Inserisci il nome utente"),
#     ],
# }

st.title("Creazione di un contatore in avanti e indietro")
st.write("Usa i bottoni per incrementare o decrementare il contatore.")

# Visualizza il contatore
st.write(f"Contatore: {st.session_state.counter}")
st.button(" incrementa + ", on_click=increment)
st.button(" decrementa - ", on_click=decrement)

# st.write('Last Updated = ', st.session_state.last_updated)

# pg = st.navigation(pages)
# pg.run()