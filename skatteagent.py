# SkatteretAssistant.py
# Streamlit-app som bruger OpenAI Assistants API med Vector Store til at besvare skatteretlige spørgsmål
# Med indbygget logfunktion til at gemme og genindlæse samtaler

import os
import streamlit as st
import uuid
import time
from datetime import datetime
import json
from openai import OpenAI
import pandas as pd
import glob
import logging
import re
import shutil

# Konfiguration
st.set_page_config(page_title="Skatteret Assistant", layout="wide")

# Konfigurer logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konstanter
ASSISTANT_ID = "asst_gknNNm2uyfxPyuzxx0JHfhtF"
VECTOR_STORE_ID = "vs_67d1e99c789c8191bd776ac5437cbc08"
PROMPTS_DIR = "prompts"
LOGS_DIR = "logs"  # Ny mappe til samtalelogfiler

# Sørg for at nødvendige mapper eksisterer
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialisering af session state
if 'assistant_id' not in st.session_state:
    st.session_state.assistant_id = None
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = None
if 'run_id' not in st.session_state:
    st.session_state.run_id = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'file_ids' not in st.session_state:
    st.session_state.file_ids = []
if 'assistant_files' not in st.session_state:
    st.session_state.assistant_files = []
if 'system_prompts' not in st.session_state:
    st.session_state.system_prompts = {}
if 'active_prompt' not in st.session_state:
    st.session_state.active_prompt = None
if 'token_count' not in st.session_state:
    st.session_state.token_count = {"input": 0, "output": 0, "total": 0}
if 'enable_web_browsing' not in st.session_state:
    st.session_state.enable_web_browsing = False
if 'original_assistant_id' not in st.session_state:
    st.session_state.original_assistant_id = None
# Nye session state variabler til logfunktion
if 'log_id' not in st.session_state:
    st.session_state.log_id = None
if 'conversation_title' not in st.session_state:
    st.session_state.conversation_title = None
if 'saved_conversations' not in st.session_state:
    st.session_state.saved_conversations = []
if 'is_loaded_conversation' not in st.session_state:
    st.session_state.is_loaded_conversation = False

# Funktion til at få OpenAI client
def get_openai_client():
    """Henter OpenAI client med API-nøgle fra miljøvariabel"""
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API-nøgle er ikke tilgængelig i miljøvariablen OPENAI_API_KEY")
        
    return OpenAI(api_key=api_key)

# Funktion til at indlæse alle tilgængelige prompts
def load_available_prompts():
    """Indlæser alle prompts fra prompt-biblioteket"""
    prompts = {}
    
    # Sørg for at mappen eksisterer
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    
    # Find alle JSON-filer i prompt-mappen
    prompt_files = glob.glob(os.path.join(PROMPTS_DIR, "*.json"))
    
    for file_path in prompt_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                
            # Tilføj filinformation
            prompt_id = os.path.basename(file_path).split('.')[0]
            prompt_data['id'] = prompt_id
            
            prompts[prompt_id] = prompt_data
        except Exception as e:
            st.error(f"Fejl ved indlæsning af prompt fra {file_path}: {e}")
    
    return prompts

# Funktion til at generere den samlede system prompt
def generate_system_instructions():
    """Genererer system prompt baseret på den aktive prompt"""
    if not st.session_state.active_prompt or st.session_state.active_prompt not in st.session_state.system_prompts:
        return ""  # Ingen aktiv prompt
    
    # Hent den aktive prompt
    prompt_data = st.session_state.system_prompts[st.session_state.active_prompt]
    
    # Returner indholdet
    return prompt_data.get('content', '')

# Funktion til at oprette eller opdatere assistent
def create_or_update_assistant(client, assistant_id, enable_web_browsing=False):
    """Opretter eller opdaterer en assistent med de ønskede værktøjer"""
    try:
        # Hent den eksisterende assistent
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        
        # Find de eksisterende værktøjer
        existing_tools = assistant.tools
        
        # Find tool typer
        existing_tool_types = [tool.type for tool in existing_tools]
        
        # Tjek om vi skal tilføje eller fjerne web_browsing
        if enable_web_browsing and "web_browsing" not in existing_tool_types:
            # Tilføj web_browsing til værktøjerne
            new_tools = existing_tools + [{"type": "web_browsing"}]
            
            # Opdater assistenten
            updated_assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tools=new_tools
            )
            logger.info(f"Web browsing aktiveret for assistent {assistant_id}")
            return updated_assistant
            
        elif not enable_web_browsing and "web_browsing" in existing_tool_types:
            # Fjern web_browsing fra værktøjerne
            new_tools = [tool for tool in existing_tools if tool.type != "web_browsing"]
            
            # Opdater assistenten
            updated_assistant = client.beta.assistants.update(
                assistant_id=assistant_id,
                tools=new_tools
            )
            logger.info(f"Web browsing deaktiveret for assistent {assistant_id}")
            return updated_assistant
            
        # Ingen ændringer nødvendige
        return assistant
            
    except Exception as e:
        st.error(f"Fejl ved opdatering af assistent: {e}")
        return None

# Funktion til at hente information om eksisterende assistent
def get_assistant_info(client, assistant_id):
    """Henter information om en eksisterende assistant"""
    try:
        assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
        return assistant
    except Exception as e:
        st.error(f"Fejl ved hentning af assistant information: {e}")
        return None

# Funktion til at uploade fil til OpenAI
def upload_file(client, file_path):
    """Uploader en fil til OpenAI"""
    try:
        with open(file_path, "rb") as file:
            response = client.files.create(
                file=file,
                purpose="assistants"
            )
        return response
    except Exception as e:
        st.error(f"Fejl ved upload af fil: {e}")
        return None

# Funktion til at oprette en thread
def create_thread(client):
    """Opretter en ny thread"""
    try:
        thread = client.beta.threads.create()
        return thread
    except Exception as e:
        st.error(f"Fejl ved oprettelse af thread: {e}")
        return None

# Funktion til at tilføje besked til thread
def add_message_to_thread(client, thread_id, content):
    """Tilføjer en besked til en thread"""
    try:
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content
        )
        return message
    except Exception as e:
        st.error(f"Fejl ved tilføjelse af besked til thread: {e}")
        return None

# Funktion til at køre assistenten
def run_assistant(client, thread_id, assistant_id):
    """Kører assistenten på en thread"""
    try:
        # Generer system prompt
        instructions = generate_system_instructions()
        
        # Tillægsparametre hvis der er en aktiv prompt
        kwargs = {}
        if instructions:
            kwargs["instructions"] = instructions
        
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            **kwargs
        )
        return run
    except Exception as e:
        st.error(f"Fejl ved kørsel af assistent: {e}")
        return None

# Funktion til at hente run status
def get_run_status(client, thread_id, run_id):
    """Henter status for en run"""
    try:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        return run
    except Exception as e:
        st.error(f"Fejl ved hentning af run status: {e}")
        return None

# Funktion til at opdatere token-tæller
def update_token_count(run):
    """Opdaterer token-tæller baseret på en run"""
    try:
        # Log run-objekt for fejlfinding
        logger.info(f"Run objekt modtaget: {type(run)}")
        
        # Tjek først efter usage attribut
        if hasattr(run, 'usage') and run.usage:
            logger.info(f"Usage fundet: {run.usage}")
            try:
                if hasattr(run.usage, 'prompt_tokens'):
                    st.session_state.token_count["input"] += run.usage.prompt_tokens
                    logger.info(f"Input tokens tilføjet: {run.usage.prompt_tokens}")
                
                if hasattr(run.usage, 'completion_tokens'):
                    st.session_state.token_count["output"] += run.usage.completion_tokens
                    logger.info(f"Output tokens tilføjet: {run.usage.completion_tokens}")
                    
                st.session_state.token_count["total"] = (
                    st.session_state.token_count["input"] + 
                    st.session_state.token_count["output"]
                )
            except Exception as usage_error:
                logger.error(f"Fejl ved behandling af usage: {usage_error}")
        
        # Alternativ metode for at finde token-forbrug
        elif hasattr(run, 'usage_metadata'):
            logger.info(f"usage_metadata fundet: {run.usage_metadata}")
            try:
                if hasattr(run.usage_metadata, 'prompt_tokens'):
                    st.session_state.token_count["input"] += run.usage_metadata.prompt_tokens
                    logger.info(f"Input tokens tilføjet: {run.usage_metadata.prompt_tokens}")
                
                if hasattr(run.usage_metadata, 'completion_tokens'):
                    st.session_state.token_count["output"] += run.usage_metadata.completion_tokens
                    logger.info(f"Output tokens tilføjet: {run.usage_metadata.completion_tokens}")
                    
                st.session_state.token_count["total"] = (
                    st.session_state.token_count["input"] + 
                    st.session_state.token_count["output"]
                )
            except Exception as metadata_error:
                logger.error(f"Fejl ved behandling af usage_metadata: {metadata_error}")
        
        # Tredje forsøg - direkte attributter
        else:
            logger.info("Forsøger at finde direkte attributter")
            tokens_updated = False
            
            if hasattr(run, 'prompt_tokens'):
                st.session_state.token_count["input"] += run.prompt_tokens
                logger.info(f"Direkte input tokens tilføjet: {run.prompt_tokens}")
                tokens_updated = True
                
            if hasattr(run, 'completion_tokens'):
                st.session_state.token_count["output"] += run.completion_tokens
                logger.info(f"Direkte output tokens tilføjet: {run.completion_tokens}")
                tokens_updated = True
                
            if tokens_updated:
                st.session_state.token_count["total"] = (
                    st.session_state.token_count["input"] + 
                    st.session_state.token_count["output"]
                )
            else:
                logger.warning("Kunne ikke finde token information i run-objektet")
                
    except Exception as e:
        logger.error(f"Kunne ikke opdatere token-tæller: {e}")
        import traceback
        logger.error(f"Fejldetaljer: {traceback.format_exc()}")

# Funktion til at hente beskeder fra thread
def get_messages(client, thread_id):
    """Henter alle beskeder fra en thread"""
    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        return messages
    except Exception as e:
        st.error(f"Fejl ved hentning af beskeder: {e}")
        return None

# Forsøg på at tilføje fil til assistent - med fallback
def add_file_to_assistant(client, assistant_id, file_id):
    """Forsøger forskellige metoder til at tilføje fil til assistent"""
    try:
        # Prøv standardmetoden først
        try:
            response = client.beta.assistants.files.create(
                assistant_id=assistant_id,
                file_id=file_id
            )
            return response
        except Exception as e1:
            st.warning(f"Standard metode til at tilføje fil fejlede: {e1}")
            
            # Prøv alternativ metode
            try:
                # Tjek om file_attachments er tilgængelig
                if hasattr(client.beta.assistants, "file_attachments"):
                    response = client.beta.assistants.file_attachments.create(
                        assistant_id=assistant_id,
                        file_id=file_id
                    )
                    return response
            except Exception as e2:
                st.warning(f"Alternativ metode også fejlet: {e2}")
                
            # Returner et mock-objekt, så applikationen kan fortsætte
            st.info("Kunne ikke tilføje fil automatisk. Tilføj filen manuelt i OpenAI dashboard.")
            class MockResponse:
                def __init__(self):
                    self.id = f"manual_{file_id}"
            return MockResponse()
            
    except Exception as e:
        st.error(f"Fejl ved tilføjelse af fil: {e}")
        return None

# Forsøg på at hente filer for en assistent - med fallback
def get_assistant_files(client, assistant_id):
    """Forsøger forskellige metoder til at hente filer for en assistent"""
    try:
        # Prøv standardmetoden først
        try:
            files = client.beta.assistants.files.list(
                assistant_id=assistant_id
            )
            return files
        except Exception as e1:
            # Returner et tomt objekt, så applikationen kan fortsætte
            class EmptyFiles:
                def __init__(self):
                    self.data = []
            return EmptyFiles()
            
    except Exception as e:
        st.error(f"Fejl ved hentning af filer: {e}")
        # Returner et tomt objekt i stedet for None
        class EmptyFiles:
            def __init__(self):
                self.data = []
        return EmptyFiles()

# Forsøg på at slette en fil fra en assistent - med fallback
def delete_file_from_assistant(client, assistant_id, file_id):
    """Forsøger forskellige metoder til at slette en fil fra en assistent"""
    try:
        # Prøv standardmetoden først
        try:
            response = client.beta.assistants.files.delete(
                assistant_id=assistant_id,
                file_id=file_id
            )
            return response
        except Exception as e1:
            st.warning(f"Standard metode til at slette fil fejlede: {e1}")
            
            # Prøv alternativ metode
            try:
                # Tjek om file_attachments er tilgængelig
                if hasattr(client.beta.assistants, "file_attachments"):
                    response = client.beta.assistants.file_attachments.delete(
                        assistant_id=assistant_id,
                        file_id=file_id
                    )
                    return response
            except Exception as e2:
                st.warning(f"Alternativ metode også fejlet: {e2}")
                
            # Returner et mock-objekt, så applikationen kan fortsætte
            st.info("Kunne ikke slette fil automatisk. Slet filen manuelt i OpenAI dashboard.")
            class MockResponse:
                def __init__(self):
                    self.id = f"deleted_{file_id}"
                    self.deleted = True
            return MockResponse()
            
    except Exception as e:
        st.error(f"Fejl ved sletning af fil: {e}")
        return None

# Funktion til at hente alle tilgængelige filer i OpenAI konto
def get_available_files(client):
    """Henter alle filer i OpenAI kontoen"""
    try:
        files = client.files.list()
        return files
    except Exception as e:
        st.error(f"Fejl ved hentning af tilgængelige filer: {e}")
        return None

# NYE FUNKTIONER TIL LOGFUNKTIONALITET

# Funktion til at generere en samtale-titel baseret på indhold
def generate_conversation_title(client, messages, max_length=8):
    """Genererer en passende titel til samtalen baseret på indholdet"""
    try:
        # Brug kun de første par beskeder for at holde token-forbrug nede
        context = []
        message_count = 0
        for msg in messages:
            if message_count >= 4:  # Brug maks 4 beskeder
                break
            context.append(f"{msg['role']}: {msg['content'][:200]}...")  # Begræns længden
            message_count += 1
        
        context_str = "\n".join(context)
        
        # Spørg OpenAI om at generere en passende titel
        prompt = f"""
        Nedenstående er en samtale om skatteret. 
        Generer en kort, beskrivende titel på maksimalt {max_length} ord, der opsummerer samtalens hovedemne.
        Du skal kun svare med titlen, intet andet.
        
        Samtale:
        {context_str}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Brug en billigere model til titlel-generering
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40,
            temperature=0.3
        )
        
        # Rens titlen
        title = response.choices[0].message.content.strip()
        
        # Fjern eventuelle citationstegn og sørg for at titlen ikke er for lang
        title = title.strip('"\'')
        
        # Begræns længden hvis nødvendigt
        words = title.split()
        if len(words) > max_length:
            title = " ".join(words[:max_length])
        
        return title
    except Exception as e:
        logger.error(f"Kunne ikke generere samtale-titel: {e}")
        # Generer en generisk titel baseret på dato og tid
        return f"Skattesamtale {datetime.now().strftime('%d-%m-%Y %H:%M')}"

# Funktion til at gemme en samtale
def save_conversation(messages, title=None, active_prompt=None):
    """Gemmer en samtale til en JSON-fil"""
    try:
        if not messages:
            st.warning("Ingen beskedhistorik at gemme.")
            return None
        
        # Generer et unikt ID til samtalen hvis ikke allerede gjort
        if not st.session_state.log_id:
            st.session_state.log_id = str(uuid.uuid4())
        
        # Brug den eksisterende titel eller generer en sikker filnavns-sikker version
        if not title:
            title = st.session_state.conversation_title or f"Samtale_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generer et filnavns-sikkert ID baseret på titlen
        safe_title = re.sub(r'[^\w\s-]', '', title).replace(' ', '_')
        file_id = f"{safe_title}_{st.session_state.log_id[:8]}"
        
        # Opret log-objektet
        log_data = {
            "id": st.session_state.log_id,
            "title": title,
            "timestamp": datetime.now().isoformat(),
            "prompt_id": active_prompt,
            "messages": messages,
            "token_count": st.session_state.token_count
        }
        
        # Definer filstien
        file_path = os.path.join(LOGS_DIR, f"{file_id}.json")
        
        # Gem til fil
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Samtale gemt til {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Fejl ved gemning af samtale: {e}")
        st.error(f"Kunne ikke gemme samtalen: {e}")
        return None

# Funktion til at indlæse en gemt samtale
def load_conversation(conversation_id):
    """Indlæser en gemt samtale fra fil"""
    try:
        # Find filstien fra ID
        file_path = None
        for log_file in glob.glob(os.path.join(LOGS_DIR, "*.json")):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
                if log_data.get("id") == conversation_id:
                    file_path = log_file
                    break
        
        if not file_path:
            st.error(f"Kunne ikke finde samtale med ID: {conversation_id}")
            return None
        
        # Indlæs samtalefilen
        with open(file_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        return conversation_data
    except Exception as e:
        logger.error(f"Fejl ved indlæsning af samtale: {e}")
        st.error(f"Kunne ikke indlæse samtalen: {e}")
        return None

# Funktion til at indlæse alle gemte samtaler
def load_all_conversations():
    """Indlæser alle gemte samtaler fra log-mappen"""
    conversations = []
    try:
        # Find alle .json filer i log-mappen
        log_files = glob.glob(os.path.join(LOGS_DIR, "*.json"))
        
        # Sorter filerne efter ændringsdato (nyeste først)
        log_files.sort(key=os.path.getmtime, reverse=True)
        
        for file_path in log_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                # Opret en forenklet repræsentation
                conversation = {
                    "id": log_data.get("id"),
                    "title": log_data.get("title", os.path.basename(file_path)),
                    "timestamp": log_data.get("timestamp", ""),
                    "message_count": len(log_data.get("messages", [])),
                    "file_path": file_path
                }
                
                # Formatér tidsstempel til et menneskelæsbart format
                try:
                    dt = datetime.fromisoformat(conversation["timestamp"])
                    conversation["display_date"] = dt.strftime("%d-%m-%Y %H:%M")
                except:
                    conversation["display_date"] = conversation["timestamp"]
                
                conversations.append(conversation)
            except Exception as e:
                logger.error(f"Fejl ved indlæsning af log-fil {file_path}: {e}")
        
    except Exception as e:
        logger.error(f"Fejl ved indlæsning af samtaler: {e}")
    
    return conversations

# Funktion til at slette en samtale
def delete_conversation(conversation_id):
    """Sletter en gemt samtale"""
    try:
        # Find filstien
        file_path = None
        for log_file in glob.glob(os.path.join(LOGS_DIR, "*.json")):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    if log_data.get("id") == conversation_id:
                        file_path = log_file
                        break
            except:
                continue
        
        if not file_path:
            st.error(f"Kunne ikke finde samtale med ID: {conversation_id}")
            return False
        
        # Slet filen
        os.remove(file_path)
        logger.info(f"Samtale slettet: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Fejl ved sletning af samtale: {e}")
        st.error(f"Kunne ikke slette samtalen: {e}")
        return False

# Hovedsiden
def main():
    st.title("Skatteretlig Assistant")
    
    # Indlæs tilgængelige prompts
    if not st.session_state.system_prompts:
        st.session_state.system_prompts = load_available_prompts()
    
    # Indlæs gemte samtaler
    if not st.session_state.saved_conversations:
        st.session_state.saved_conversations = load_all_conversations()
    
    # Verificer API-nøgle
    try:
        client = get_openai_client()
        st.sidebar.success("OpenAI API-nøgle er tilgængelig fra miljøvariablen")
    except ValueError as e:
        st.error(str(e))
        st.info("API-nøglen er ikke konfigureret i miljøvariablen OPENAI_API_KEY")
        return
    
    # Debug information i sidebar
    with st.sidebar:
        st.header("Information")
        st.write(f"**Vector Store ID:** {VECTOR_STORE_ID}")
        st.write(f"**Assistant ID:** {ASSISTANT_ID}")
        
        # Vis token-tæller
        st.header("Token Forbrug")
        col_tokens1, col_tokens2 = st.columns(2)
        with col_tokens1:
            st.metric("Input tokens", st.session_state.token_count['input'])
            st.metric("Output tokens", st.session_state.token_count['output'])
        with col_tokens2:
            st.metric("Total tokens", st.session_state.token_count['total'])
            # Tilføj estimeret pris (baseret på o3-mini priser - $1.10 per 1M tokens input, $4.40 per 1M tokens output)
            input_cost = st.session_state.token_count['input'] * 1.10 / 1000000
            output_cost = st.session_state.token_count['output'] * 4.40 / 1000000
            total_cost = input_cost + output_cost
            st.metric("Est. omkostning ($)", round(total_cost, 6))
        
        # Web browsing omkostninger
        if st.session_state.enable_web_browsing:
            st.warning("Web browsing aktiveret - ekstra omkostning på $1 per 1000 søgninger")
        
        # Flueben til at aktivere web browsing
        st.header("Funktioner")
        enable_web = st.checkbox("Aktiver web browsing", value=st.session_state.enable_web_browsing)
        
        # Opdater web browsing indstilling hvis ændret
        if enable_web != st.session_state.enable_web_browsing:
            with st.spinner("Opdaterer assistent konfiguration..."):
                if st.session_state.assistant_id:
                    # Gem original assistent ID hvis det ikke allerede er gemt
                    if not st.session_state.original_assistant_id:
                        st.session_state.original_assistant_id = st.session_state.assistant_id
                    
                    # Opdater assistenten
                    updated_assistant = create_or_update_assistant(
                        client, 
                        st.session_state.assistant_id, 
                        enable_web
                    )
                    
                    if updated_assistant:
                        # Opdater session state
                        st.session_state.enable_web_browsing = enable_web
                        st.success(f"Web browsing {'aktiveret' if enable_web else 'deaktiveret'}")
                        st.rerun()  # Genindlæs siden for at vise ændringerne
    
    # Opret to kolonner i layoutet
    col1, col2 = st.columns([1, 2])
    
    # Venstre kolonne: Administration og tidligere samtaler
    with col1:
        # Vis faner for administration og logfiler
        tab1, tab2 = st.tabs(["Administration", "Tidligere samtaler"])
        
        # Tab 1: Administration
        with tab1:
            st.header("Administration")
            
            st.info(f"Vector Store ID: {VECTOR_STORE_ID}")
            
            # Hent information om predefineret assistent
            if not st.session_state.assistant_id:
                with st.spinner("Henter assistent information..."):
                    assistant = get_assistant_info(client, ASSISTANT_ID)
                    if assistant:
                        st.session_state.assistant_id = assistant.id
                        st.success(f"Forbundet til assistent: {assistant.name}")
                        
                        # Opret også en thread
                        thread = create_thread(client)
                        if thread:
                            st.session_state.thread_id = thread.id
                            st.success(f"Thread oprettet med ID: {thread.id}")
                    else:
                        st.error(f"Kunne ikke hente assistent med ID: {ASSISTANT_ID}")
                        st.info("Kontroller at assistent-ID'et er korrekt og at din API-nøgle har adgang.")
            
            # Viser ID'er hvis assistenten er aktiv
            if st.session_state.assistant_id:
                with st.expander("Assistant og Thread information", expanded=True):
                    assistant_info = get_assistant_info(client, st.session_state.assistant_id)
                    if assistant_info:
                        st.write(f"**Assistant navn:** {assistant_info.name}")
                        st.write(f"**Model:** {assistant_info.model}")
                        st.write(f"**Assistant ID:** {assistant_info.id}")
                        st.write(f"**Thread ID:** {st.session_state.thread_id or 'Ikke oprettet'}")
                        st.write(f"**Vector Store ID:** {VECTOR_STORE_ID}")
                        
                        # Vis tools og konfiguration
                        if hasattr(assistant_info, 'tools') and assistant_info.tools:
                            st.write("**Aktiverede værktøjer:**")
                            for tool in assistant_info.tools:
                                st.write(f"- {tool.type}")
                        
                        # Vis instructions hvis tilgængelige
                        if hasattr(assistant_info, 'instructions') and assistant_info.instructions:
                            st.markdown("**Assistent instruktioner:**")
                            st.markdown(f"```\n{assistant_info.instructions}\n```")
                    else:
                        st.write(f"**Assistant ID:** {st.session_state.assistant_id}")
                        st.write(f"**Thread ID:** {st.session_state.thread_id or 'Ikke oprettet'}")
                
                # Nulstil samtale
                with st.expander("Nulstil samtale"):
                    if st.button("Start ny samtale"):
                        # Gem den aktuelle samtale hvis der er beskeder
                        if st.session_state.messages:
                            # Generer en titel til samtalen hvis der ikke allerede er en
                            if not st.session_state.conversation_title and len(st.session_state.messages) > 1:
                                with st.spinner("Genererer samtale-titel..."):
                                    title = generate_conversation_title(client, st.session_state.messages)
                                    st.session_state.conversation_title = title
                            
                            # Gem samtalen
                            with st.spinner("Gemmer den aktuelle samtale..."):
                                save_conversation(
                                    st.session_state.messages, 
                                    st.session_state.conversation_title,
                                    st.session_state.active_prompt
                                )
                                # Opdater listen over gemte samtaler
                                st.session_state.saved_conversations = load_all_conversations()
                        
                        # Opret ny thread
                        thread = create_thread(client)
                        if thread:
                            st.session_state.thread_id = thread.id
                            st.session_state.messages = []
                            st.session_state.log_id = None
                            st.session_state.conversation_title = None
                            st.session_state.is_loaded_conversation = False
                            st.success("Ny samtale startet")
                            st.rerun()
            
            # Vis prompt-liste
            st.header("Prompts")
            
            if not st.session_state.system_prompts:
                st.info("Ingen prompts fundet i 'prompts' mappen.")
            else:
                # Lav en liste med prompt-navne
                prompt_options = [(prompt_id, prompt_data.get('title', prompt_id)) 
                                for prompt_id, prompt_data in st.session_state.system_prompts.items()]
                
                # Tilføj "Ingen prompt" mulighed
                prompt_options.insert(0, (None, "Ingen prompt"))
                
                # Sorter alfabetisk efter titel (men behold "Ingen prompt" øverst)
                prompt_options[1:] = sorted(prompt_options[1:], key=lambda x: x[1])
                
                # Opret en radio-knap for hver prompt
                selected_option = st.radio(
                    "Vælg en prompt til assistenten:",
                    options=[prompt_id for prompt_id, _ in prompt_options],
                    format_func=lambda x: next((title for pid, title in prompt_options if pid == x), "Ingen prompt"),
                    index=0 if st.session_state.active_prompt is None else 
                        next((i for i, (pid, _) in enumerate(prompt_options) if pid == st.session_state.active_prompt), 0)
                )
                
                # Opdater den aktive prompt
                if selected_option != st.session_state.active_prompt:
                    st.session_state.active_prompt = selected_option
                    
                    # Hvis en prompt er valgt, vis dens indhold
                    if selected_option:
                        with st.expander("Vis prompt indhold", expanded=False):
                            prompt_data = st.session_state.system_prompts[selected_option]
                            prompt_content = prompt_data.get('content', 'Intet indhold')
                            st.text_area("Prompt indhold", prompt_content, disabled=True, height=250)
            
        # Tab 2: Tidligere samtaler
        with tab2:
            st.header("Gemte samtaler")
            
            # Genindlæs listen over gemte samtaler
            if st.button("Opdater liste"):
                st.session_state.saved_conversations = load_all_conversations()
                st.success("Listen over gemte samtaler er opdateret")
            
            # Vis liste over gemte samtaler
            if not st.session_state.saved_conversations:
                st.info("Ingen gemte samtaler fundet.")
            else:
                st.write(f"Fandt {len(st.session_state.saved_conversations)} gemte samtaler:")
                
                # Opret en container til at vise samtalerne
                saved_chat_container = st.container()
                with saved_chat_container:
                    # Vis hver samtale
                    for idx, conversation in enumerate(st.session_state.saved_conversations):
                        with st.expander(f"{conversation['title']} ({conversation['display_date']})"):
                            st.write(f"**Dato:** {conversation['display_date']}")
                            st.write(f"**Antal beskeder:** {conversation['message_count']}")
                            
                            col_conv1, col_conv2 = st.columns([3, 1])
                            with col_conv1:
                                # Knap til at indlæse samtalen
                                if st.button("Indlæs samtale", key=f"load_{idx}", 
                                             use_container_width=True):
                                    with st.spinner("Indlæser samtale..."):
                                        # Indlæs samtaledata
                                        conv_data = load_conversation(conversation["id"])
                                        
                                        if conv_data:
                                            # Gem den aktuelle samtale hvis der er beskeder og ikke en indlæst samtale
                                            if (st.session_state.messages and 
                                                not st.session_state.is_loaded_conversation and
                                                st.session_state.log_id != conv_data["id"]):
                                                
                                                # Spørg om at gemme den aktuelle samtale
                                                if st.session_state.messages:
                                                    # Generer en titel til samtalen hvis der ikke allerede er en
                                                    if not st.session_state.conversation_title:
                                                        title = generate_conversation_title(
                                                            client, 
                                                            st.session_state.messages
                                                        )
                                                        st.session_state.conversation_title = title
                                                    
                                                    # Gem samtalen
                                                    save_conversation(
                                                        st.session_state.messages, 
                                                        st.session_state.conversation_title,
                                                        st.session_state.active_prompt
                                                    )
                                            
                                            # Indlæs den valgte samtale
                                            st.session_state.messages = conv_data["messages"]
                                            st.session_state.log_id = conv_data["id"]
                                            st.session_state.conversation_title = conv_data["title"]
                                            st.session_state.is_loaded_conversation = True
                                            
                                            # Indlæs også token-tæller hvis tilgængelig
                                            if "token_count" in conv_data:
                                                st.session_state.token_count = conv_data["token_count"]
                                            
                                            # Indlæs prompt hvis tilgængelig
                                            if "prompt_id" in conv_data and conv_data["prompt_id"]:
                                                st.session_state.active_prompt = conv_data["prompt_id"]
                                            
                                            st.success(f"Samtale '{conv_data['title']}' indlæst")
                                            st.rerun()
                            
                            with col_conv2:
                                # Knap til at slette samtalen
                                if st.button("Slet", key=f"delete_{idx}", use_container_width=True):
                                    if delete_conversation(conversation["id"]):
                                        st.session_state.saved_conversations = load_all_conversations()
                                        st.success("Samtale slettet")
                                        st.rerun()
            
            # Knap til at gemme den aktuelle samtale
            if st.session_state.messages:
                st.header("Gem aktuel samtale")
                
                # Vis nuværende titel hvis tilgængelig
                if st.session_state.conversation_title:
                    st.write(f"Aktuel titel: **{st.session_state.conversation_title}**")
                
                # Indtast en alternativ titel
                custom_title = st.text_input(
                    "Indtast en titel til samtalen (valgfrit):",
                    value=st.session_state.conversation_title or ""
                )
                
                if st.button("Gem samtalen nu", use_container_width=True):
                    with st.spinner("Gemmer samtalen..."):
                        # Brug custom_title hvis indtastet, ellers generer en titel
                        title_to_use = custom_title
                        if not title_to_use:
                            title_to_use = generate_conversation_title(client, st.session_state.messages)
                            
                        # Opdater titlen i session state
                        st.session_state.conversation_title = title_to_use
                        
                        # Gem samtalen
                        file_path = save_conversation(
                            st.session_state.messages, 
                            title_to_use,
                            st.session_state.active_prompt
                        )
                        
                        if file_path:
                            # Opdater listen over gemte samtaler
                            st.session_state.saved_conversations = load_all_conversations()
                            st.success(f"Samtale gemt som '{title_to_use}'")
                            st.rerun()
    
    # Højre kolonne: Chat
    with col2:
        st.header("Skatteret Rådgivning")
        
        # Først tjek om vi er forbundet til assistenten
        if not st.session_state.assistant_id:
            st.warning("Venter på forbindelse til assistenten...")
        else:
            # Vis velkomstbesked hvis ingen beskeder findes
            if not st.session_state.messages:
                assistant_info = get_assistant_info(client, st.session_state.assistant_id)
                if assistant_info and hasattr(assistant_info, 'name'):
                    st.info(f"Velkommen til {assistant_info.name}. Stil dit spørgsmål om skattelovgivning nedenfor.")
                else:
                    st.info("Velkommen til Skatteret Assistant. Stil dit spørgsmål om skattelovgivning nedenfor.")
            
            # Vis om dette er en indlæst samtale
            if st.session_state.is_loaded_conversation:
                st.success(f"Du arbejder med en indlæst samtale: '{st.session_state.conversation_title}'")
            
            # Vis chathistorik
            chat_container = st.container(height=400)
            with chat_container:
                # Vis eksisterende beskeder
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.chat_message("user").write(message["content"])
                    else:
                        st.chat_message("assistant").write(message["content"])
            
            # Input felt til nye spørgsmål
            if st.session_state.thread_id:
                # Lav en chat input
                prompt = st.chat_input("Stil et spørgsmål om skatteret...")
                
                if prompt:
                    # Vis brugerens spørgsmål
                    st.chat_message("user").write(prompt)
                    
                    # Gem brugerens besked
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    # Tilføj besked til thread
                    with st.spinner("Sender dit spørgsmål..."):
                        message = add_message_to_thread(client, 
                                                     st.session_state.thread_id, 
                                                     prompt)
                    
                    # Kør assistenten
                    with st.spinner("Assistenten behandler dit spørgsmål..."):
                        run = run_assistant(client, 
                                         st.session_state.thread_id, 
                                         st.session_state.assistant_id)
                        
                        if run:
                            st.session_state.run_id = run.id
                            
                            # Vent på at kørslen er færdig
                            while True:
                                run_status = get_run_status(client, 
                                                         st.session_state.thread_id, 
                                                         st.session_state.run_id)
                                
                                # Vis status for web-søgning hvis det er aktiveret
                                if st.session_state.enable_web_browsing and run_status.status == "in_progress":
                                    # Tjek status på tool calls
                                    if hasattr(run_status, 'required_action') and run_status.required_action:
                                        if run_status.required_action.type == "submit_tool_outputs":
                                            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                                                if tool_call.type == "web_browsing" or "web" in tool_call.type.lower():
                                                    # Vis info om web søgning
                                                    st.info(f"🌐 Assistenten søger på nettet for information...")
                                
                                if run_status.status == "completed":
                                    # Forsøg at hente token-brug
                                    try:
                                        # Hent den fulde run-status for at sikre alle attributter
                                        complete_run = client.beta.threads.runs.retrieve(
                                            thread_id=st.session_state.thread_id,
                                            run_id=st.session_state.run_id
                                        )
                                        
                                        # Prøv at hente token-brug - tjek om API'et understøtter usage
                                        try:
                                            # Tjek om dette API er tilgængeligt
                                            has_usage_api = (hasattr(client.beta.threads.runs, 'usage') and 
                                                           callable(getattr(client.beta.threads.runs.usage, 'list')))
                                            
                                            if has_usage_api:
                                                usage_response = client.beta.threads.runs.usage.list(
                                                    thread_id=st.session_state.thread_id,
                                                    run_id=st.session_state.run_id
                                                )
                                                logger.info(f"Usage response: {usage_response}")
                                                
                                                # Behandle usage_response hvis data er tilgængelig
                                                if hasattr(usage_response, 'data') and usage_response.data:
                                                    for usage_item in usage_response.data:
                                                        if hasattr(usage_item, 'prompt_tokens'):
                                                            st.session_state.token_count["input"] += usage_item.prompt_tokens
                                                        if hasattr(usage_item, 'completion_tokens'):
                                                            st.session_state.token_count["output"] += usage_item.completion_tokens
                                                    
                                                    st.session_state.token_count["total"] = (
                                                        st.session_state.token_count["input"] + 
                                                        st.session_state.token_count["output"]
                                                    )
                                        except Exception as usage_error:
                                            logger.error(f"Fejl ved hentning af usage: {usage_error}")
                                            # Fortsæt med at bruge complete_run hvis usage API fejler
                                            update_token_count(complete_run)
                                            
                                        # Hvis vi ikke fik opdateret token-tælleren fra usage API
                                        if "total" not in st.session_state.token_count or st.session_state.token_count["total"] == 0:
                                            update_token_count(complete_run)
                                        
                                    except Exception as e:
                                        logger.error(f"Kunne ikke hente token information: {e}")
                                        st.warning("Kunne ikke opdatere token-tæller for denne forespørgsel")
                                    break
                                elif run_status.status == "requires_action":
                                    # Håndter tool calls, herunder web_browsing
                                    if hasattr(run_status, 'required_action') and run_status.required_action:
                                        if run_status.required_action.type == "submit_tool_outputs":
                                            # Behandl web browsing eller andre tool outputs her
                                            st.info("Assistenten udfører handlinger...")
                                            
                                            # Automatisk handling for tool outputs
                                            tool_outputs = []
                                            for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                                                # For web browsing behøver vi ikke at gøre noget, OpenAI håndterer det
                                                tool_outputs.append({
                                                    "tool_call_id": tool_call.id,
                                                    "output": "Forespørgsel behandlet automatisk"
                                                })
                                            
                                            # Submit tool outputs
                                            client.beta.threads.runs.submit_tool_outputs(
                                                thread_id=st.session_state.thread_id,
                                                run_id=st.session_state.run_id,
                                                tool_outputs=tool_outputs
                                            )
                                elif run_status.status in ["failed", "cancelled", "expired"]:
                                    st.error(f"Fejl ved kørsel: {run_status.status}")
                                    if hasattr(run_status, 'last_error'):
                                        st.error(f"Fejldetaljer: {run_status.last_error}")
                                    break
                                
                                # Vent lidt før næste check
                                time.sleep(1)
                    
                    # Hent svar
                    with st.spinner("Henter svar..."):
                        messages = get_messages(client, st.session_state.thread_id)
                        
                        if messages and messages.data:
                            # Nyeste besked først
                            latest_message = messages.data[0]
                            
                            if latest_message.role == "assistant":
                                # Vis assistentens svar
                                content = latest_message.content[0].text.value
                                st.chat_message("assistant").write(content)
                                
                                # Gem assistentens besked
                                st.session_state.messages.append({"role": "assistant", "content": content})
                                
                                # Vis citations hvis de findes
                                if hasattr(latest_message, "annotations") and latest_message.annotations:
                                    st.info("Dette svar indeholder citations fra dokumenter i Vector Store og/eller filbilag.")
                                
                                # Hvis dette er første svar i samtalen, generer en titel
                                if len(st.session_state.messages) == 2 and not st.session_state.conversation_title:
                                    with st.spinner("Genererer samtale-titel..."):
                                        title = generate_conversation_title(client, st.session_state.messages)
                                        st.session_state.conversation_title = title
                                        st.success(f"Samtale navngivet: {title}")
                                
                                # Autogem samtalen efter hver udveksling hvis den ikke allerede er gemt
                                if st.session_state.log_id is None:
                                    with st.spinner("Gemmer samtalen automatisk..."):
                                        save_conversation(
                                            st.session_state.messages, 
                                            st.session_state.conversation_title,
                                            st.session_state.active_prompt
                                        )
                                        # Opdater listen over gemte samtaler
                                        st.session_state.saved_conversations = load_all_conversations()
                                else:
                                    # Opdater den eksisterende samtale-fil
                                    with st.spinner("Opdaterer gemt samtale..."):
                                        save_conversation(
                                            st.session_state.messages, 
                                            st.session_state.conversation_title,
                                            st.session_state.active_prompt
                                        )
            else:
                st.warning("Venter på oprettelse af samtale-tråd...")

if __name__ == "__main__":
    main()