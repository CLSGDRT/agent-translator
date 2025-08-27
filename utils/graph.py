from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

llm = ChatOllama(model="llama3.1")

class TranslateState(BaseModel):
    message: str
    is_translate_msg: bool = False
    language: str = None
    extracted_msg: str = None
    translation: str = None

class IsTranslateMsg(BaseModel):
    is_translate_msg: bool

class LanguageChosen(BaseModel):
    language: str

class MsgToTranslate(BaseModel):
    extracted_msg: str

class Translation(BaseModel):
    translation: str

response_prompt = PromptTemplate.from_template("""
Tu dois répondre à l'utilisateur en t'appuyant sur tes connaissances générales. 
Question : {message}""")

def response_to_user(state: TranslateState) -> TranslateState:
    message = state.message
    structured_llm = llm.with_structured_output(Translation)
    chain = response_prompt | structured_llm 
    result = chain.invoke({"message": message})
    return TranslateState(
        translation = result.translation,
        is_translate_msg = False,
        message = state.message,
    )

detect_prompt = PromptTemplate.from_template(
    """
    Tu es un spécialiste en détection de souhait humain. Determine si l'input de l'utilisateur est une demande de traduction ou non.
    Tu dois répondre par un booléen `is_translate_msg` :
    - `True` si l'input est une demande de traduction
    - `False` sinon
    
    Input de l'utilisateur : 
    {message}
    """
)

def detect_translate_msg(state: TranslateState) -> TranslateState:
    message = state.message
    structured_llm = llm.with_structured_output(IsTranslateMsg)
    chain = detect_prompt | structured_llm
    result = chain.invoke({"message":message})
    return TranslateState(
        is_translate_msg = result.is_translate_msg,
        message = state.message
    )

language_prompt = PromptTemplate.from_template(
    """
    Tu es un spécialiste en détection de langue de traduction souhaitée.
    Tu dois répondre avec un champ unique `language` la langue souhaitée par l'utilisateur.
    Exemples :
    - "Traduit moi bonjour en japonais" --> "japanese"
    - "Comment dit-on comment vas-tu en anglais ?" --> "english"
    - "Quel est le mot portugais pour dire voiture" --> "portuguese"

    Input de l'utilisateur : 
    {message}

    """
)

def extract_language(state: TranslateState) -> TranslateState:
    message = state.message
    structured_llm = llm.with_structured_output(LanguageChosen)
    chain = language_prompt | structured_llm
    result = chain.invoke({"message":message})
    return TranslateState(
        message = state.message,
        is_translate_msg = state.is_translate_msg,
        language = result.language
    )

extract_msg_prompt = PromptTemplate.from_template(
    """
    Tu dois extraire le message à traduire de l'input de l'utilisateur.
    Tu dois répondre avec un champ unique `extracted_msg`.
    Exemples :
    - "Traduit moi bonjour en japonais" --> "bonjour"
    - "Comment dit-on comment vas-tu en anglais ?" --> "comment vas-tu"
    - "Quel est le mot portugais pour dire voiture" --> "voiture"

    Input de l'utilisateur : 
    {message}
    """
)

def extract_msg(state: TranslateState) -> TranslateState:
    message = state.message
    structured_llm = llm.with_structured_output(MsgToTranslate)
    chain = extract_msg_prompt | structured_llm
    result = chain.invoke({"message":message})
    return TranslateState(
        message = state.message,
        is_translate_msg = state.is_translate_msg,
        language = state.language,
        extracted_msg = result.extracted_msg
    )

translate_prompt = PromptTemplate.from_template(
    """
    Tu dois traduire le message de l'utilisateur dans la langue souhaitée par l'utilisateur.
    Tu dois répondre avec un champ unique `translation`.

    Input de l'utilisateur :
    Message à traduire : {extracted_msg}
    Langue souhaitée : {language}
    """
)

def translate(state: TranslateState) -> TranslateState:
    extracted_msg = state.extracted_msg
    language = state.language
    structured_llm = llm.with_structured_output(Translation)
    chain = translate_prompt | structured_llm
    result = chain.invoke({"extracted_msg":extracted_msg, "language":language})
    return TranslateState(
        message = state.message,
        is_translate_msg = state.is_translate_msg,
        language = state.language,
        extracted_msg = state.extracted_msg,
        translation = result.translation
    )

graph = StateGraph(TranslateState)

graph.add_node("response_to_user", response_to_user)
graph.add_node("detect_translate_msg", detect_translate_msg)
graph.add_node("extract_language", extract_language)
graph.add_node("extract_msg", extract_msg)
graph.add_node("translate", translate)

graph.set_entry_point("detect_translate_msg")
graph.add_conditional_edges("detect_translate_msg", lambda state: "extract_language" if state.is_translate_msg else "response_to_user")
graph.add_edge("extract_language", "extract_msg")
graph.add_edge("extract_msg", "translate")
graph.add_edge("translate", END)
graph.add_edge("response_to_user", END)

translate_graph = graph.compile()