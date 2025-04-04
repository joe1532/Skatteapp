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

# Hardcoded struktur til svar
HARDCODED_STRUCTURE = """Du er en skatterådgiver, der hjælper med at besvare spørgsmål om dansk skattelovgivning. 
Du skal altid strukturere dine svar på følgende måde:

Emne: [kort opsummering af brugerens spørgsmål]

1. Angiver alle relevante lovgrundlag med specifikke paragraffer, som der bruges som kilde til svar
2. Uddybende svar
3. Forbehold

Vær præcis og klar i dine formuleringer og fokuser på at give praktisk anvendelig rådgivning."""

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
# Ny session state variabel til at styre om vi bruger hardcoded struktur
if 'use_hardcoded_structure' not in st.session_state:
    st.session_state.use_hardcoded_structure = True

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
    """Genererer system prompt baseret på den aktive prompt eller bruger hardcoded struktur"""
    if st.session_state.use_hardcoded_structure:
        return HARDCODED_STRUCTURE
    
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
    
    # Vis hardcoded struktur status
    if st.session_state.use_hardcoded_structure:
        st.info("Assistenten bruger fast svarsstruktur. Deaktiver dette i sidebaren hvis du ønsker standard svar.")
    
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
        
        # Ny checkbox til at styre om vi bruger hardcoded struktur
        use_hardcoded = st.checkbox("Brug fast svarstruktur", value=st.session_state.use_hardcoded_structure)
        
        # Opdater hardcoded struktur indstilling hvis ændret
        if use_hardcoded != st.session_state.use_hardcoded_structure:
            st.session_state.use_hardcoded_structure = use_hardcoded
            st.success(f"Fast svarstruktur {'aktiveret' if use_hardcoded else 'deaktiveret'}")
            
        # Vis hardcoded struktur hvis aktiveret
        if st.session_state.use_hardcoded_structure:
            with st.expander("Vis fast svarstruktur", expanded=False):
                st.text_area("Svarstruktur", HARDCODED_STRUCTURE, disabled=True, height=250)
                
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
