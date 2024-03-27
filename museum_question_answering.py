from transformers import AutoTokenizer, AutoModelForSequenceClassification, WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
from TTS.utils.synthesizer import Synthesizer
import requests
from datasets import load_dataset

def download_file(url):
    local_filename = url.split('/')[-1] #last part of filename
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_filename

def transcribe_audio(processor, transcription_model, audio_sample):
    input_features = processor(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt").input_features
    predicted_ids = transcription_model.generate(input_features) # generate token ids
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True) #decode into text
    print(transcription)

def is_question(nlp, text):
    classification = nlp(text)
    if classification['label']=='LABEL_1':
        print("Question, confidence=", classification['score'])
    else:
        print("Statement, confidence=", classification['score'])


"""
Answer generation (question text -> answer text)'
Can add context of the painting TIAGo is next to. This helps if the client is asking a question that only refers in the abstract e.g. "tell me about THIS painting"
Using Falcon7B
"""

def falcon_text(text_generation_pipeline, tokenizer, question, context=''):
    role = 'You are an eccentric British professor of art history, answer the following question '
    if context != '':
        context = 'You are next to the painting ' + context
    prompt = context + role + question
    sequences = text_generation_pipeline(
        prompt,
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def synthesize(david_voice, text):
    wav = david_voice.tts(text)
    return wav


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    #LOAD MODELS
    # load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    transcription_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    transcription_model.config.forced_decoder_ids = None

    question_tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    question_detection_model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    nlp = pipeline("text-classification", model=question_detection_model, tokenizer=question_tokenizer)#, aggregation_strategy='average') #'max'

    answer_generation_model = "tiiuae/falcon-7b-instruct"
    answer_tokenizer = AutoTokenizer.from_pretrained(answer_generation_model)
    text_generation_pipeline = pipeline(
        "text-generation",
        model=answer_generation_model,
        tokenizer=answer_tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    #model (David Attenborough) loaded from hugging face https://huggingface.co/enlyth/baj-tts/
    model_file = '/content/david.pth'
    config_path = '/content/config.json'

    david_voice = Synthesizer(
        tts_config_path=config_path,
        tts_checkpoint=model_file,
        use_cuda=bool(torch.cuda.is_available()),
    )


    #TESTING SAMPLE:
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = ds[27]["audio"]

    transcribed_text = transcribe_audio(processor, transcription_model, audio_sample)
    if is_question(nlp, transcribed_text):
        falcon_text(text_generation_pipeline, answer_tokenizer, transcribed_text, context='Starry Night')
        output_audio_wav = synthesize(david_voice, "this is AI David Attenborough speaking")
        david_voice.save_wav(output_audio_wav, '/content/speech_output.wav')

if __name__ == "__main__":
  main()
