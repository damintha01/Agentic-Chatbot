from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os
import re


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id, to_front=False):

    # Prevent the same thread from being added multiple times
    if thread_id in st.session_state["chat_threads"]:

        if to_front:
            st.session_state["chat_threads"].remove(thread_id)
            st.session_state["chat_threads"].insert(0, thread_id)

    else:

        if to_front:
            st.session_state["chat_threads"].insert(0, thread_id)

        else:
            st.session_state["chat_threads"].append(thread_id)


def get_first_user_message(thread_id):

    messages = load_conversation(thread_id)

    for message in messages:
        if isinstance(message, HumanMessage):
            content = str(message.content).strip()

            if content:
                return content

    return ""


def build_topic_title(source_text):

    cleaned_text = " ".join(str(source_text).split()).strip()

    if not cleaned_text:
        return "New Chat "

    generated_title = ""

    try:
        import backend as backend_module

        generated_title = " ".join(
            str(backend_module.generate_thread_title(cleaned_text)).split()
        ).strip()

    except Exception:
        generated_title = ""

    if generated_title:

        normalized_title = generated_title.lower().strip(" .,:;!?\"'")

        if normalized_title.startswith("about "):
            generated_title = generated_title[6:].strip()

        if normalized_title in {"this", "about", "new chat"}:
            generated_title = ""

        elif len(generated_title.split()) == 1 and len(generated_title) < 4:
            generated_title = ""

    if generated_title:
        return generated_title

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "be",
        "can",
        "could",
        "do",
        "for",
        "give",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "please",
        "show",
        "tell",
        "the",
        "to",
        "what",
        "where",
        "with",
        "would",
        "you",
    }

    words = re.findall(r"[A-Za-z0-9]+", cleaned_text.lower())
    meaningful_words = [word for word in words if word not in stop_words]

    if not meaningful_words:
        meaningful_words = words[:4]

    if not meaningful_words:
        return "New Chat"

    return " ".join(meaningful_words[:4]).title()


def get_thread_title(thread_id, source_text=None):

    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    cached_title = st.session_state["thread_titles"].get(thread_id)

    if cached_title:
        return cached_title

    title_source = source_text or get_first_user_message(thread_id)

    if not title_source:
        title = "New Chat"

    else:
        title = build_topic_title(title_source)

    st.session_state["thread_titles"][thread_id] = title
    return title


def update_thread_title(thread_id, source_text):

    if not source_text:
        return get_thread_title(thread_id)

    if "thread_titles" not in st.session_state:
        st.session_state["thread_titles"] = {}

    current_title = st.session_state["thread_titles"].get(thread_id)

    if current_title and current_title != "New Chat":
        return current_title

    title = build_topic_title(source_text)

    st.session_state["thread_titles"][thread_id] = title
    return title


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"], to_front=True)

    # Reset the visible title for the fresh thread
    st.session_state["thread_titles"][st.session_state["thread_id"]] = "New Chat"


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...",
                    expanded=True
                )
            }

            def resumed_ai_only_stream():

                # Resume the graph with the human decision
                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):

                    # Update tool execution status
                    if isinstance(
                        message_chunk,
                        ToolMessage
                    ):

                        tool_name = getattr(
                            message_chunk,
                            "name",
                            "tool"
                        )

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                    # Stream only assistant-generated text
                    if isinstance(
                        message_chunk,
                        AIMessage
                    ):

                        if message_chunk.content:
                            yield message_chunk.content

            # Display the streamed final answer
            resumed_ai_message = st.write_stream(
                resumed_ai_only_stream()
            )

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(
                interrupted_thread_id
            )

            if next_interrupt is not None:

                save_pending_interrupt(
                    interrupted_thread_id,
                    next_interrupt
                )

                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete",
                    expanded=False
                )

            else:

                # No more pending approval
                st.session_state["pending_hitl"] = None

                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete",
                    expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic Chatbot with LangGraph",
    page_icon="🤖"
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top, rgba(22, 70, 44, 0.28) 0%, rgba(7, 12, 10, 0.1) 28%, rgba(3, 5, 4, 1) 65%),
                linear-gradient(180deg, #050706 0%, #030403 100%);
            color: #eef7ef;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07120d 0%, #040706 100%);
            border-right: 1px solid rgba(72, 210, 124, 0.18);
        }

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: #e5f7e8;
        }

        .app-shell {
            max-width: 1240px;
            margin: 0 auto;
            padding: 0.75rem 0 2rem;
        }

        .app-hero {
            padding: 1rem 0 1.4rem;
            margin-bottom: 1rem;
        }

        .app-eyebrow,
        .sidebar-eyebrow {
            color: #2fe17a;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }

        .app-hero h1 {
            color: #f2faf3;
            font-size: clamp(2.8rem, 5vw, 4.8rem);
            line-height: 0.98;
            margin: 0;
            text-shadow: 0 0 28px rgba(45, 224, 118, 0.16);
        }

        .app-subtitle {
            color: rgba(218, 238, 222, 0.7);
            font-size: 1.05rem;
            line-height: 1.6;
            max-width: 58rem;
            margin-top: 0.95rem;
        }

        section[data-testid="stSidebar"] button,
        .stButton button {
            border-radius: 14px;
            border: 1px solid rgba(80, 216, 128, 0.22);
            background: linear-gradient(180deg, rgba(19, 31, 24, 0.95) 0%, rgba(9, 14, 11, 0.98) 100%);
            color: #effaf0;
            transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
        }

        section[data-testid="stSidebar"] button:hover,
        .stButton button:hover {
            border-color: rgba(80, 216, 128, 0.55);
            box-shadow: 0 0 0 1px rgba(80, 216, 128, 0.15), 0 10px 30px rgba(0, 0, 0, 0.28);
            transform: translateY(-1px);
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(72, 210, 124, 0.12);
            border-radius: 20px;
            background: linear-gradient(180deg, rgba(10, 14, 12, 0.96) 0%, rgba(7, 9, 8, 0.96) 100%);
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.22);
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            color: #f0f7f1;
        }

        div[data-testid="stChatMessage"] svg,
        div[data-testid="stChatMessage"] span {
            color: #3bf08b;
        }

        [data-testid="stChatInput"] {
            border-top: 1px solid rgba(72, 210, 124, 0.16);
            backdrop-filter: blur(10px);
        }

        [data-testid="stChatInput"] textarea {
            background: rgba(8, 12, 9, 0.92) !important;
            color: #f3fbf4 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="app-shell">
        <div class="app-hero">
            <div class="app-eyebrow">Enterprise Conversational Workspace</div>
            <h1>Agentic Chatbot with LangGraph</h1>
            <div class="app-subtitle">
                Dark operational interface with green system accents, topic-based conversation history,
                and tool-aware assistance.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Clear the composer text before the widget is instantiated this run.
# (Setting st.session_state["composer_text"] directly after the text_area
# widget has been created raises a StreamlitAPIException, so a flag is used
# to defer the reset to the start of the next run.)
if st.session_state.get("clear_composer_text"):
    st.session_state["composer_text"] = ""
    st.session_state["clear_composer_text"] = False


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# Store the sidebar titles for conversations
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# =============================================================


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"], to_front=True)


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar threading feature =========================

st.sidebar.markdown(
    """
    <div class="sidebar-eyebrow">Workspace</div>
    <h2>My Conversations</h2>
    <p>Newest threads appear first. Each entry is labeled by topic instead of a raw ID.</p>
    """,
    unsafe_allow_html=True
)


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


# Display all conversation threads from newest to oldest
for thread_id in st.session_state["chat_threads"]:

    thread_label = get_thread_title(thread_id)

    if len(thread_label) > 34:
        thread_label = f"{thread_label[:31].rstrip()}..."

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        thread_label,
        key=thread_id,
        type="primary" if thread_id == st.session_state["thread_id"] else "secondary",
        use_container_width=True
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # ========================= HITL ADDED =========================

        # Restore any pending approval for this conversation
        sync_pending_interrupt(thread_id)

        # =============================================================

        # Rerun the application to display the loaded messages
        st.rerun()


# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])


# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get(
    "pending_hitl"
)

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


# Display approval controls
if current_thread_has_pending_hitl:

    st.warning(
        "🧑 Human approval required\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    # Approve button
    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")

    # Reject button
    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            # Send "no" back to interrupt()
            resume_hitl_execution("no")



# ========================= Visible prompt composer =========================

st.markdown(
    "<div class='app-eyebrow' style='margin-top: 2rem;'>Prompt Composer</div>",
    unsafe_allow_html=True
)

user_input = st.text_area(
    "Message",
    placeholder="Type your prompt here",
    height=92,
    key="composer_text",
    label_visibility="collapsed",
    disabled=current_thread_has_pending_hitl,
)

uploaded_file = st.file_uploader(
    "Attach a PDF",
    type=["pdf"],
    key="composer_file",
    label_visibility="collapsed",
    disabled=current_thread_has_pending_hitl,
)

send_clicked = st.button(
    "Send message",
    type="primary",
    use_container_width=True,
    disabled=current_thread_has_pending_hitl,
)


# Process the uploaded PDF if one was attached
if send_clicked and uploaded_file:

    # Store the temporary file path
    temporary_file_path = None

    try:

        # Save the uploaded PDF as a temporary local file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getvalue()
            )

            temporary_file_path = temporary_file.name

        # Call the existing backend RAG ingestion function
        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):

            ingest_rag_document(
                temporary_file_path
            )

        # Display PDF processing confirmation
        st.toast(
            f"{uploaded_file.name} processed successfully.",
            icon="✅"
        )

    except Exception as error:

        # Display PDF processing error
        st.error(
            f"PDF processing failed: {error}"
        )

    finally:

        # Delete the temporary PDF after indexing
        if (
            temporary_file_path
            and os.path.exists(temporary_file_path)
        ):
            os.remove(temporary_file_path)


# Default user input value
if send_clicked:
    user_input = user_input.strip()

    if not user_input:
        st.session_state["clear_composer_text"] = True
        st.rerun()


# Run this block after the user submits a text message
if send_clicked and user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):

        # Use a mutable holder so the generator can set/modify it
        status_holder = {
            "box": None
        }

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Lazily create & update the SAME status container
                # when any tool runs
                if isinstance(
                    message_chunk,
                    ToolMessage
                ):

                    tool_name = getattr(
                        message_chunk,
                        "name",
                        "tool"
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …",
                            expanded=True
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(
                    message_chunk,
                    AIMessage
                ):
                    yield message_chunk.content

            # ========================= HITL ADDED =========================

            # interrupt() pauses the graph without returning
            # a completed ToolMessage.
            #
            # Inspect the saved checkpoint after streaming ends.
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )

            if pending_interrupt is not None:

                # Save the interrupt for displaying approval buttons
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )

                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase "
                    "button below."
                )

            # =============================================================

        ai_message = st.write_stream(
            ai_only_stream()
        )

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:

            # Check whether execution is waiting for approval
            if get_pending_interrupt(
                st.session_state["thread_id"]
            ) is not None:

                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete",
                    expanded=False
                )

            else:

                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete",
                    expanded=False
                )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # Clear the visible composer after sending
    st.session_state["clear_composer_text"] = True
    st.session_state.pop("composer_file", None)

    # Promote the thread label from a placeholder to a topic title
    update_thread_title(
        st.session_state["thread_id"],
        user_input
    )

    # ========================= HITL ADDED =========================

    # Rerun so the composer clears and, if applicable, approval controls
    # (rendered earlier in the script) appear immediately after interrupt().
    st.rerun()

    # =============================================================