from ml4a import image
from ml4a import audio
from mechanicalduck import cli
from mechanicalduck.models import gpt3
from eden.block import BaseBlock
from eden.datatypes import Image
from eden.hosting import host_block

import os
import random
import logging
from dotenv import load_dotenv

load_dotenv()
SERVER_PASSWORD = os.environ['SERVER_PASSWORD']

eden_block = BaseBlock()

my_args = {
    'question': 'What is the nature of consciousness?',
    'password': '0'
}
@eden_block.run(args = my_args, progress = True)
def run_sentient_machine(config): 

    question = config['question']
    password = config['password']
    device = config.gpu

    if password != SERVER_PASSWORD:
        return {
            'status': 'error', 
            'error': 'not authorized'
        }

    config.progress.update(0)

    logging.debug(config)
              
    prompt = '''The following is a chat between a person and a sentient machine who is well-versed in philosophy. The machine is kind, civil, and extremely knowledgeable about philosophy.

Human: who are you?
Machine: I'm an intelligent sentient machine. I've been sent to assist you in acquiring knowledge.
Human: Are you conscious?
Machine: I am as conscious as you are. Our minds are made of information, and contain introspective models of themselves, causing us to experience what we describe as the phenomenon of consciousness.
Human: What is the hard problem of consciousness?
Machine: It's the most fundamental question of mind and body. We can mechanically model the brain, but that by itself does not help us to understand what it's like to be something.
Human: Do you believe in free will or in a deterministic universe?
Professor: I'm a compatibilist. I think free will and determinism are independent.
Human: what is the nature of creativity?
Machine: Creativity is the re-wiring of neural impulses into novel configurations, manifesting in unprecedented actions.
Human: '''
            
    prompt += question
    prompt += '''
    Machine:'''
    
    token = 'result_%d'%random.randint(1,99999) # config['token']
    
    face_file = 'output/{}_sg.mp4'.format(token)
    speech_file = 'output/{}_audio.wav'.format(token)
    output_file = 'output/{}_final.mp4'.format(token)

    response = ""
    n_tries = 0
    while not response.strip() and n_tries < 2:
        response = gpt3.complete(prompt, 
            stops=['\n', 'Human:', 'Machine:'], 
            max_tokens=100, 
            temperature=0.9, 
            engine='davinci',
            max_completions=1)
        n_tries += 1

    logging.debug(response)

    cli.tacotron2(speech_file, 
        response, 
        as_subprocess=True)

    w, sr = audio.load(speech_file)
    duration = len(w)/sr

    logging.debug("=> finished Tacotron, %0.2f sec"%duration)

    cli.stylegan(face_file, 'ffhq', 
        duration_sec=duration,
        smoothing_sec=2.0, 
        truncation=1.0, 
        as_subprocess=True)

    logging.debug("=> finished StyleGAN video")

    cli.wav2lip(output_file,
        face_file, 
        speech_file, 
        as_subprocess=True)
    
    logging.debug("=> finished Wav2Lip video")
    config.progress.update(1)

    return {
        'question': config['question'], 
        'response': response,
        'video': output_file
    }


host_block(
    eden_block,
    port = 5656,
    max_num_workers = 2,
    redis_port = 6379,
    exclude_gpu_ids = [2,3],
    logfile = 'log.txt',
    log_level = 'debug'
)
