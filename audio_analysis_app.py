#Audio Analysis

#Install streamlit library
#pip install streamlit

#Install libraries/packages
'''
pip install pydub
pip install SpeechRecognition
pip install happytransformer
pip install language-tool-python
pip install textblob
'''

# Import necessary libraries
import streamlit as st
from pydub import AudioSegment
import io
import os
import speech_recognition as sr
from happytransformer import HappyTextToText, TTSettings
import pandas as pd
import numpy as np
from language_tool_python import LanguageTool
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import cmudict
from textblob import TextBlob
import math


# Function to convert audio file to WAV format (adapted for Streamlit)
def convert_to_wav(input_file):
    try:
        audio_data = input_file.read()  # Read the binary data from the uploaded file
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data)) 
        output_wav = io.BytesIO()
        audio_segment.export(output_wav, format='wav')
        st.write(f"Uploaded audio file has been converted to WAV format.")
        return output_wav
    except Exception as e:
        st.error("Failed to convert the audio file to WAV format")
        st.error(f'Error processing audio file: {e}')
        return None

# Function to transcribe audio file
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.warning(f'No speech detected in the uploaded audio')
        return None
    except sr.RequestError as e:
        st.error(f'Could not request results from Google Speech Recognition service; {e}')
        return None
    

# Function to generate suggested formal texts
def suggested_texts(original_text):
    from happytransformer import HappyTextToText, TTSettings

    #Initialize the HappyTextToText model
    happy_c_t_f = HappyTextToText("T5", "prithivida/informal_to_formal_styletransfer")

    #Define the settings for text generation
    top_k_sampling_settings = TTSettings(do_sample=True, top_k=20, temperature=0.7, min_length=1, max_length=200)
   
    suggested_texts_list = []
    
    for i in range(3): #Generate 3 suggested texts
        
        result = happy_c_t_f.generate_text(original_text, args=top_k_sampling_settings)
        suggested_text = result.text
        print(f'Suggested text {i+1}:')
        print(suggested_text)
        print()
        suggested_texts_list.append(suggested_text)
    
    return suggested_texts_list

# Function to compare original text with suggested text and provide suggestions
def compare_and_suggest(original_text, suggested_text):
    original_text_lower = original_text.lower()
    suggested_text_lower = suggested_text.lower()

    original_words = original_text_lower.split()
    suggested_words = suggested_text_lower.split()

    #Calculate missing words
    missing_words = [word for word in original_words if word not in suggested_words]

    if len(original_words) > 0:
        missing_percentage = (len(missing_words) / len(original_words)) * 100
    else:
        missing_percentage = 0

    #Determine the nature of errors or needed corrections
    if original_text_lower == suggested_text_lower:
       suggestion = "The transcript appears clear and well-spoken."
    else:
        # Check for errors indicating potential mistakes
        if missing_percentage > 50:
            suggestion = "The transcript shows significant errors that may require corrections."
        elif missing_percentage > 20:
            suggestion = "There are errors in the transcript that suggest minor discrepancies or needed corrections."
        elif missing_percentage > 5:
            suggestion = "Minor errors are observed in the transcript, indicating areas for improvement."
        else:
            suggestion = "The transcript is nearly perfect, with only slight improvements needed."

    print(f"\nSuggestion: {suggestion}")

    return suggestion

# Function to assess grammar and provide suggestions for a given transcription
def assess_grammar(transcription):
    # Initialize LanguageTool for grammar checking
    tool = LanguageTool('en-IN')
    
    if pd.isna(transcription):
        return np.nan, 'No speech detected'

    grammar_matches = [match for match in tool.check(transcription) if match.ruleIssueType == 'grammar' and 'hyphen' not in match.message.lower()]
    num_errors = len(grammar_matches)
    #suggestions = ",".join([match.message for match in grammar_matches])
    suggestions = ",".join([match.message
                            .replace('‘', "'")
                            .replace('’', "'")
                            .replace('“', '"')
                            .replace('”', '"')
                            for match in grammar_matches])
    
    if num_errors == 0:
        return 0, None
    
    return num_errors, suggestions


# Function to score the transcription based on grammar errors
def score_grammar(transcription):
    if pd.isna(transcription):
        return np.nan
    num_errors, _ = assess_grammar(transcription)
    tokens = word_tokenize(str(transcription))
    score = 1 - (num_errors / len(tokens))
    return score

# Function to calculate TTR (Type-Token Ratio) for a given transcription
def calculate_ttr(transcription):
    if pd.isna(transcription):
        return np.nan, 'No speech detected'
    tokens = word_tokenize(transcription.lower())
    total_words = len(tokens)
    unique_words = len(set(tokens))
    ttr = unique_words / total_words

    if ttr < 0.2:
        suggestion = "Try using a wider range of vocabulary and synonyms to increase lexical diversity."
    elif ttr > 0.8:
        suggestion = "Your vocabulary diversity is impressive! Make sure the varied vocabulary enhances clarity."
    else:
        suggestion = "Your vocabulary diversity is good. Keep exploring new words to enrich your language."
    return ttr, suggestion

# Function to analyze pronunciation score for a given transcription
def pronunciation_score(transcription):
    #Load the CMU Pronunciation Dictionary
    nltk.download('cmudict')
    phonetic_dict = cmudict.dict()
    
    if pd.isna(transcription):
        return np.nan
    tokens = word_tokenize(transcription)
    total_score = 0
    total_tokens = 0

    for token in tokens:
        phenomes = phonetic_dict.get(token.lower())
        if phenomes:
            total_score += len(phenomes[0])
            total_tokens += 1

    if total_tokens > 0:
        average_score = total_score / total_tokens
        normalized_score = min(1, average_score / 10)
        return normalized_score
    else:
        return 0

# Function to provide suggestions for improving pronunciation based on the score
def pronunciation_suggestions(pronunciation_score):
    if math.isnan(pronunciation_score):
        return 'No speech detected'        
    elif pronunciation_score < 0.2:
       return "You need more practice and attention to improve your pronunciation. Focus on practicing individual sounds"
    elif pronunciation_score < 0.5:
       return "Keep practicing to refine your pronunciation. Practice saying words slowly and clearly, focusing on how you stress syllables"
    elif pronunciation_score < 0.8:
       return "You're doing well, but keep practicing to improve tricky sounds and speak more clearly."
    else:
       return "Your pronunciation is excellent! Keep it up!"
   
# Function to analyze fluency based on subjectivity score
def fluency_analysis(transcription):
    if pd.isna(transcription):
        return np.nan
    blob = TextBlob(transcription)
    subjectivity_score = blob.sentiment.subjectivity
    return subjectivity_score

# Function to provide suggestions for improving fluency based on the analysis
def fluency_suggestions(fluency_score):
    if math.isnan(fluency_score):
        return 'No speech detected'
    elif fluency_score < 0.2:
       return "Practice speaking more slowly and clearly to improve fluency."
    elif fluency_score < 0.5:
       return "Work on expressing your thoughts more confidently and fluidly for better fluency."
    elif fluency_score < 0.8:
       return "You're expressing your ideas well, but aim for more natural and cohesive speech."
    else:
       return "Your fluency and coherence are excellent! Keep it up!"
   
# Function to calculate coherence score based on sentence sentiment
def coherence_score(transcription):
    if pd.isna(transcription):
        return np.nan
    sentences = sent_tokenize(transcription)
    sentiments = [TextBlob(sentence).sentiment.polarity for sentence in sentences]
    normalized_sentiments = [(sentiment + 1) / 2 for sentiment in sentiments]
    coherence_score = sum(normalized_sentiments) / len(normalized_sentiments)
    return coherence_score

# Function to provide suggestions for improving coherence based on the coherence score
def coherence_suggestions(coherence_score):
    if pd.isna(coherence_score):
        return "No speech detected."

    if coherence_score <= 0.2:
        return "The text lacks coherence and logical connections between ideas. Consider restructuring sentences and paragraphs to improve flow."
    elif coherence_score <= 0.5:
        return "There are some logical connections between ideas, but the text could benefit from clearer transitions and organization."
    elif coherence_score <= 0.9:
        return "The text demonstrates good coherence overall, with clear connections between ideas. Review and refine transitions for even better flow."
    else:
        return "The text exhibits excellent coherence and logical flow, with seamless transitions between ideas. Well done!"

# Function to perform overall analysis
def analyze_transcription(transcription):
    
    if transcription is None:
        return {
            #'Filename': os.path.basename(input_file),
            'Speech_detected': False,
            'Note': 'No speech detected.',
            'Transcriptions': None,
            'Suggested Texts': [],
            'Grammar Errors': None,
            'Grammar Suggestions': None,
            'Grammar Score': None,
            'Vocabulary Score': None,
            'Vocabulary Suggestions': None,
            'Pronunciation Score': None,
            'Pronunciation Suggestions': None,
            'Fluency Score': None,
            'Fluency Suggestions': None,
            'Coherence Score': None,
            'Coherence Suggestions': None
        }

    #Generate suggested texts
    suggested_transcripts = suggested_texts(transcription)
    suggestion = compare_and_suggest(transcription, suggested_transcripts[0])
    
    #Grammar analysis
    grammar_errors, grammar_suggestions = assess_grammar(transcription)
    grammar_score = score_grammar(transcription)
    
    #Vocabulary analysis
    vocabulary_score, vocabulary_suggestions = calculate_ttr(transcription)
    
    #Pronunciation analysis
    pronunciation = pronunciation_score(transcription)
    pronunciation_suggestion = pronunciation_suggestions(pronunciation)
    fluency_score = fluency_analysis(transcription)
    fluency_suggestion = fluency_suggestions(fluency_score)
    coherence = coherence_score(transcription)
    coherence_suggestion = coherence_suggestions(coherence)
    
    return {
        #'Filename': filename,
        'Speech_detected': True,
        'Transcriptions': transcription,
        'Note': " Variation in the transcript can be attributed to the clarity in the speech. If the transcript seems to have a lot of variations, it is suggested to improve the clarity of the speech.",
        'Suggested Texts': suggested_transcripts,
        'Grammar Errors': grammar_errors,
        'Grammar Suggestions': suggestion if grammar_suggestions is None else [suggestion, grammar_suggestions],
        'Grammar Score': grammar_score,
        'Vocabulary Score': vocabulary_score,
        'Vocabulary Suggestions': vocabulary_suggestions,
        'Pronunciation Score': pronunciation,
        'Pronunciation Suggestions': pronunciation_suggestion,
        'Fluency Score': fluency_score,
        'Fluency Suggestions': fluency_suggestion,
        'Coherence Score': coherence,
        'Coherence Suggestions': coherence_suggestion,
        }
    

# Streamlit code integrated within the same script
def main():
    st.title('Audio Analysis')

    # File upload widget
    uploaded_file = st.file_uploader("Upload an audio file")

    if uploaded_file:
        #Convert audio file to wav format
        wav_file = convert_to_wav(uploaded_file)
        
        if wav_file:
            #Transcribe the converted wav file
            transcript_text = transcribe_audio(wav_file)
            
            if transcript_text:
                st.subheader("Transcript from the audio file:")
                st.write(transcript_text)
                st.write("Note: Variation in the transcript can be attributed to the clarity in the speech. If the transcript seems to have a lot of variations, it is suggested to improve the clarity of the speech.")
                
                #Perform analysis
                analysis_result = analyze_transcription(transcript_text)
                
                # Display analysis results
                st.subheader('Analysis Results:')

                #st.subheader('Suggested Texts:')
                for i, text in enumerate(analysis_result['Suggested Texts']):
                    st.write(f"Suggested Text {i + 1}: {text}")

                st.subheader('Suggestions:')
                st.write(f"Grammar: {analysis_result['Grammar Suggestions']}")
                st.write(f"Vocabulary: {analysis_result['Vocabulary Suggestions']}")
                st.write(f"Pronunciation: {analysis_result['Pronunciation Suggestions']}")
                st.write(f"Fluency: {analysis_result['Fluency Suggestions']}")
                st.write(f"Logical Consistency: {analysis_result['Coherence Suggestions']}")
                
            else:
                st.warning("Failed to transcribe the audio file")

if __name__ == '__main__':
    main()


