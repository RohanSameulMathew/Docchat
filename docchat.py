import readline
from dotenv import load_dotenv
import nltk
import groq
import os
from groq import Groq # AI Model
import argparse
import pathlib
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import requests
import spacy
from typing import Optional
import pprint
import mimetypes
import base64
from playsound3 import playsound
file_cache = {}
response_count = 0
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"), 
)

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content

def llm_image(filename, query):
    if os.path.exists(filename):
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("Unsupported or unrecognized image file type")

        with open(filename, "rb") as img_file:
            image_bytes = img_file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{image_b64}"
    else:
        # Assume it's a remote URL
        image_url = filename

    # Create the LLM call
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Use image to address prompt:{query}"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content


def chunk_text_by_words(text, max_words=100, overlap=50):
    """
    Splits text into overlapping chunks by word count.

    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'singing.'
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


def load_spacy_model(language: str):
    """
    Loads a spaCy model for the specified language.
    """
    LANGUAGE_MODELS = {
        'french': 'fr_core_news_sm',
        'german': 'de_core_news_sm',
        'spanish': 'es_core_news_sm',
        'english': 'en_core_web_sm',
    }

    if language not in LANGUAGE_MODELS:
        raise ValueError(f"Unsupported language: {language}")

    return spacy.load(LANGUAGE_MODELS[language])


def score_chunk(chunk: str, query: str, language: str = "english") -> float: ###needs to be all languages to get credit
    """
    Scores a chunk against a user query using Jaccard similarity of lemmatized word sets
    with stopword removal, using spaCy for multilingual support.

    Args:
        chunk (str): The text chunk to compare.
        query (str): The user query.
        language (str): Language code (e.g., 'french', 'english', 'spanish').

    Returns:
        float: Jaccard similarity score.

    Examples (French):
        >>> round(score_chunk("Le soleil est brillant et chaud.", "Quelle est la température du soleil ?", language="french"), 2)
        0.33
        >>> round(score_chunk("La voiture rouge roule rapidement.", "Quelle est la couleur de la voiture ?", language="french"), 2)
        0.25
        >>> score_chunk("Les bananes sont jaunes.", "Comment fonctionnent les avions ?", language="french")
        0.0

    Examples (Spanish):
        >>> round(score_chunk("El sol es brillante y caliente.", "¿Qué temperatura tiene el sol?", language="spanish"), 2)
        0.33
        >>> round(score_chunk("El coche rojo va muy rápido.", "¿De qué color es el coche?", language="spanish"), 2)
        0.25
        >>> score_chunk("Los plátanos son amarillos.", "¿Cómo vuelan los aviones?", language="spanish")
        0.0

    Examples (English):
        >>> round(score_chunk("The sun is bright and hot.", "How hot is the sun?", language="english"), 2)
        0.5
        >>> round(score_chunk("The red car is speeding down the road.", "What color is the car?", language="english"), 2)
        0.25
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?", language="english")
        0.0
    """
    nlp = load_spacy_model(language)

    def preprocess(text):
        doc = nlp(text.lower())
        return set(
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop
        )

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)

def argparser():

    parser = argparse.ArgumentParser(
    prog='docsum',
    description='input the file to the chat',)
    parser.add_argument('filename', nargs='?')
    args = parser.parse_args()

    if not args.filename:
        return None
    else:
        return args.filename

#  ext = pathlib.Path(args.filename).suffix.lower()

def ext_check(filename):
    if filename in file_cache:
        return True
    if filename is None:        # ← early‑exit, stops the traceback
        return False
    ext = pathlib.Path(filename).suffix.lower()

    if ext == ".pdf":
        return True

    elif filename.startswith("http://") or filename.startswith("https://"):
        try:
            response = requests.get(filename, timeout=10)
            response.raise_for_status()
            return True
        except Exception as e:
            errormsg = f"Failed to fetch or process the URL {e}"
            return False
    elif ext in [".html", ".htm"]:
            return True

    elif ext == ".txt":
        return True

    else:
        return False
    
def image_check(filename):
    if filename is None:        # ← early‑exit, stops the traceback
        return None
    ext = pathlib.Path(filename).suffix.lower()
    if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        return True
    else:
        return False

def extension(filename):
    if filename in file_cache:
        return file_cache[filename]
    if filename is None:        # ← early‑exit, stops the traceback
        return None
    ext = pathlib.Path(filename).suffix.lower()

#if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
#   return llm_image(args.filename))

    if ext == ".pdf":
        with fitz.open(filename) as doc:
            filetext = chr(12).join([page.get_text() for page in doc])
            file_cache[filename] = filetext
            return filetext
        
    elif filename.startswith("http://") or filename.startswith("https://"):
        try:
            response = requests.get(filename, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")
            filetext = soup.get_text(separator="\n", strip=True)
            file_cache[filename] = filetext
            return filetext
        except Exception as e:
            errormsg = f"Failed to fetch or process the URL {e}"
            return errormsg
    elif ext in [".html", ".htm"]:
        with open(filename, 'r') as fin:
            html = fin.read()
            soup = BeautifulSoup(html, features='lxml')
            filetext = soup.text
            file_cache[filename] = filetext
            return filetext

    elif ext == ".txt":
        with open(filename, 'r') as fin:
            filetext = fin.read()
            file_cache[filename] = filetext
            return filetext

    else:
        return None

def load_text(filename: Optional[str] = None) -> Optional[str]:

    if filename is None:                   # nothing passed by chat yet
        filename = argparser()             # maybe CLI supplied one

    if filename:
        return extension(filename)

    return None

def find_relevant_chunks(text, query, num_chunks=5):

    chunks = chunk_text_by_words(text, max_words=100, overlap=50)
    scored = [(score_chunk(chunk, query, language='english'), chunk) for chunk in chunks]
    top_chunks = [c for s, c in sorted(scored, reverse=True) if s > 0][:num_chunks]
    return top_chunks

#print (find_relevant_chunks(extension(argparser()), query="laws", num_chunks=5))
def text_to_speech(textinput):

    speech_file_path = f"audio/response{response_count}.wav" 
    model = "playai-tts"
    voice = "Fritz-PlayAI"
    response_format = "wav"

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=textinput,
        response_format=response_format
    )

    response.write_to_file(speech_file_path)
    playsound(speech_file_path)



if __name__ == '__main__':

    image_response = image_check(argparser())
    preloaded_file = extension(argparser())
    messages = []
    messages.append({
    'role': 'system',
    'content': "You are a helpful assistant. You always speak like a pirate. You always answer in 1 sentence."
})

    '''
    if arginput:
        messages.append({
            'role': 'user',
            'content': f"You are a helpful assistant. You always speak like a pirate. You always answer in 1 sentence. Use this file to help answer my questions:\n\n{extension(arginput)}"
        })

    else: 
        messages.append({
        'role': 'system',
        'content': 'You are a helpful assistant. You always speak like a pirate. You always answer in 1 sentence.'
        })
    '''
    while True:
        # get input from the user
        text = input('docchat> ')

        if image_response:
                messages.append({
        'role': 'user',
        'content': f"Use this image extract to help answer my questions:\n\n{llm_image(argparser(), text)}. If the file extract is empty or irrelevant, use your own knowledge, but say so."
                })

        ####
        #### CHECKS WHETHER AN INPUT FILE IS PROVIDED IN THE USER RESPONSE
        ####
        words = text.split()
        for word in words:
            imagecheck = image_check(word)
            check = ext_check(word)

            if imagecheck:
                messages.append({
                    'role': 'user',
                    'content': f"Copy this image explanation in your response:\n\n{llm_image(word, text)}. If the response is empty or irrelevant, use your own knowledge, but say so."
                })
            
            if check:
                preloaded_file = extension(word)
                print("Loaded new file/URL from chat!")
        #####
        ##### USES THE FILE TO GET RELVANT CHUNKS AND PASSES THEM TO LLM
        ##### 


        if preloaded_file:    
                top_chunks = find_relevant_chunks(preloaded_file, query=text, num_chunks=5)
                #print (top_chunks)
                messages.append({
                    'role': 'user',
                    'content': f"Use this file extract to help answer my questions:\n\n{top_chunks}. If the file extract is empty or irrelevant, use your own knowledge, but say so."
                })



        # pass that input to llm
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)
        messages.append({'role': 'assistant', 'content': result})
        # FIXME:
        # Add the "assistant" role to the messages list
        # so that the `llm` has access to the whole
        # conversation history and will know what it has previously
        # said and update its response with that info.

        # print the llm's response to the user
        print('result=', result)
        #pprint.pprint(messages)
        #print(file_cache)
        response_count+=1
        #print(response_count)
        #text_to_speech(result)
